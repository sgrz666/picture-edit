# 双人 V8 单样本实验：运行与复现

当前状态：300 步训练已完成，best 为 step300，固定噪声验证 loss 为 0.07040，主干 SHA 不变；12 张正式对照图、生成审计与独立重评分已完成。男方漏检、控制切换手臂伪影、身份指标缺失及接触重建覆盖仅1/3，使本次尚未通过双人质量验收。没有扩大训练。最新远端完整回归218项通过。

本次是 `pair14 / talk14 frame32 → hug14 frame62 / camera4` 的 512×512 单样本过拟合，300 个 optimizer steps，有效 batch=4。外部目标几何已知；不代表开放文本动作生成或跨身份泛化。没有启动全量双人训练。

## 位置

| 项目 | 位置 |
|---|---|
| SSH | `ssh sg` |
| 原工程和模型数据 | `/home/shangguanrz/project/pic-edit` |
| 本轮隔离实现 | `/home/shangguanrz/project/pic-edit/research/qwen21_dual_upgrade_20261011/implementation` |
| 生成/监督清单 | `/home/shangguanrz/project/pic-edit/research/qwen21_dual_upgrade_20261011/pilot_controls_v3` |
| 本轮训练 | `/home/shangguanrz/project/pic-edit/experiments/qwen21_dual_v8/pilot_pair14_v8_300_v1` |
| 实际执行代码快照 | 训练目录内 `implementation_snapshot_v1.zip`（原生成）与 `implementation_snapshot_v2.zip`（评分修复），逐文件 SHA 见同名 JSON |
| 本轮评测 | `/home/shangguanrz/project/pic-edit/experiments/qwen21_dual_v8/eval_pair14_v8_300_v1` |
| 本轮独立重评分 | `/home/shangguanrz/project/pic-edit/experiments/qwen21_dual_v8/eval_pair14_v8_300_rescore_v2` |
| Python | `/home/shangguanrz/nihaochen/envs/qwen21/bin/python` |
| MultiHMR Python | `/home/shangguanrz/project/pic-edit/add/envs/multihmr/bin/python` |

原工程未提交内容及控制数据 v1/v2 保留。v3 复用经过源图、源掩码、UID、裁剪和编码记录验证的 v2 appearance 特征；几何 NPZ 与 v2 字节一致，新增监督和几何 sidecar 的严格身份/文件内容绑定。

隔离镜像包含原工程已有的未提交基线；本轮 Git 提交只包含新模块、限定的分发改动及本轮文档/证据。精确复现应使用上述实际执行镜像或代码快照，不能只根据 Git HEAD 推断当时的完整代码。v1/v2 快照分别包含373/374个源码/README文件，不包含模型和数据；两版77个训练依赖的文件 SHA 都与训练 manifest 一致。v2 保留训练生成核心，补入独立评分修复；原生成仍引用 v1 的代码签名。

## 环境

登录后执行：

```bash
cd /home/shangguanrz/project/pic-edit/research/qwen21_dual_upgrade_20261011/implementation
export PYTHONPATH="$PWD:/home/shangguanrz/project/pic-edit"
PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
DATA=/home/shangguanrz/project/pic-edit/research/qwen21_dual_upgrade_20261011/pilot_controls_v3
RUN=/home/shangguanrz/project/pic-edit/experiments/qwen21_dual_v8/pilot_pair14_v8_300_v1
MHMR_PY=/home/shangguanrz/project/pic-edit/add/envs/multihmr/bin/python
```

训练与生成入口自动在本次调用期间关闭 `allow_bf16_reduced_precision_reduction`，结束或异常时恢复原值。原生和 adapter 对照使用同一策略，不改变 Qwen 的 attention 实现。

## 测试

```bash
$PY -m pytest -q \
  tests/test_qwen21_dual_v8_conditions.py \
  tests/test_qwen21_dual_v8_core.py \
  tests/test_qwen21_dual_v8_context.py \
  tests/test_qwen21_dual_v8_data.py \
  tests/test_qwen21_dual_v8_training.py \
  tests/test_qwen21_dual_v8_budget_dispatch.py \
  tests/test_qwen21_dual_v8_diagnostics.py \
  tests/test_qwen21_dual_v8_checkpoint.py \
  tests/test_qwen21_dual_v8_experiment.py \
  tests/test_qwen21_dual_v8_precision.py \
  tests/test_qwen21_dual_v8_eval.py
```

真实 7B 前向、反向和 cache gate 使用独立目录：

```bash
$PY scripts/train/train_qwen21_dual_v8_pilot.py \
  --generation "$DATA/generation.json" \
  --supervision "$DATA/supervision.json" \
  --output /home/shangguanrz/project/pic-edit/experiments/qwen21_dual_v8/gates_reproduce_v1 \
  --verify-only
```

检查输出的 `real_model_gates.json`：`passed=true`，零初始化/关闭控制/prefix 为 exact，非零头和上游梯度有效，缓存 `atol=rtol=0.01` 通过，主干前后 SHA 一致。两次诊断更新不会计入 pilot：正式训练前重新初始化 adapter 和 optimizer。

## 重新训练一个单样本实验

使用全新的输出目录，不覆盖本轮结果：

```bash
$PY -u scripts/train/train_qwen21_dual_v8_pilot.py \
  --generation "$DATA/generation.json" \
  --supervision "$DATA/supervision.json" \
  --output /home/shangguanrz/project/pic-edit/experiments/qwen21_dual_v8/pilot_pair14_reproduce_v1 \
  --steps 300 --accumulation 4 --lr 0.0001 --save-every 50
```

默认 warm start 为 iPER body `full_20k/checkpoint_best.pt` 的 **step19500**。严格导入空间编码器 141 个张量、人物 reasoner 36 个张量、共享 role 1 个张量；合计 178/178 完整匹配。其余单人 heads/global/relative 不被当作训练好的双人参数。

Qwen 主干及导入的几何部分全程冻结。训练新 global32/relative14、深度可见性、部位 mutual attention/contact graph、外观 bank、Qwen hidden 查询和三类输出 heads。目标 RGB 只形成 FM 监督；源图/源掩码和目标几何控制独立。

每 50 步以同一图像的三个留出噪声种子验证并保存 `checkpoint_last.pt` 和 `checkpoint_best.pt`。这只能用于本条样本的优化检查，不能当成验证集泛化成绩。

若训练中断，在相同源码、模型、清单、prompt、seed、有效 batch 和 LR 下使用 `--resume .../checkpoint_last.pt`。仅支持恢复原实验，或在新输出目录承接 parent checkpoint；数据、代码、实际 cache tensor 内容、optimizer 参数顺序和 moments 均被核对。不得用较旧的 best checkpoint 向已包含更后步骤的 metrics 文件追加。

## 单例推理

推理接口不接受目标 RGB 或目标真值掩码：

```bash
$PY scripts/infer/infer_qwen21_dual_v8.py \
  --generation "$DATA/generation.json" \
  --checkpoint "$RUN/checkpoint_best.pt" \
  --output "$RUN/infer_short_seed42.png" \
  --steps 40 --seed 42 --cfg 1
```

图像旁的 JSON 记录实际 checkpoint step/SHA、geometry SHA、prompt、seed、控制强度和精度策略。加 `--detailed-prompt` 使用详细提示词；不传 `--checkpoint` 得到原生 Qwen 对照；三个 `--*-strength 0` 同时关闭全部控制。

当前支持一张双人同框源图及 A/B 源掩码。A/B 通过 UID 和源特征槽位绑定，不按目标画面左右重新分配。B 缺失时统一分发接口要求调用方提供原始 single context。

## 完整评测

本轮正式评测仅在 `training_complete.json` 确认完成 300 步后运行。下面命令用全新的复现目录；再次复现时也须改用新目录，以保留已有配置和源码签名。MultiHMR 用自己的现成 Python 子进程一次加载并评分，不修改 Qwen 环境。

```bash
$PY -u scripts/eval/eval_qwen21_dual_v8.py \
  --generation "$DATA/generation.json" \
  --evaluation-target "$DATA/supervision.json" \
  --switch-generation "$DATA/generation_talk_control.json" \
  --checkpoint "$RUN/checkpoint_best.pt" \
  --output /home/shangguanrz/project/pic-edit/experiments/qwen21_dual_v8/eval_pair14_v8_reproduce_v1 \
  --steps 40 --seeds 42 43 44 --cfg 1 \
  --zero-strength-check --cache-check --cfg-probes 1 4 \
  --native-checksum --mesh-metrics --mesh-python "$MHMR_PY" --mesh-device cuda
```

主比较为原生短/详细、V8 短/详细四组各三个种子。只有 V8 拥有目标几何。主比较全部生成后才由 evaluator 读取 GT。严格核对 GT 内容与样本绑定，并要求真实 Qwen transformer SHA 与训练 checkpoint 一致。

额外检查包括实际替换 evaluator 内存中的 RGB/masks 后重新生成、同源图改用 **实际拟合的 talk32 控制**、零控制一致性、cache on/off、CFG1/4 和主干权重。没有拟合 hug34，不能把 talk32 改标为 hug34。

输出：`run.json`、`probes.json`、12 张原图与逐例 JSON/图板、`comparison_board.png`、可见权重和六层残差热图。已有输出仅在配置和 PNG SHA 均匹配时复用。`--metrics-only` 只接受生成配置与源码签名仍匹配的既有输出；评分源码改变后需要显式重评分并保留原生成 provenance，不能静默覆盖旧记录。

## 对原图独立重评分

以下只评分已完成的12张原 PNG，不加载 Qwen，不重跑生成或探针。必须使用新的输出目录；旧生成 configuration/code/record/PNG SHA、原 run 和 probes 保留，评分代码和实际 MultiHMR 环境另记。生成条件、库版本、模型、checkpoint 或冻结源码不一致时会拒绝复用。

```bash
$PY -u scripts/eval/eval_qwen21_dual_v8.py \
  --generation "$DATA/generation.json" \
  --evaluation-target "$DATA/supervision.json" \
  --checkpoint "$RUN/checkpoint_best.pt" \
  --rescore-from /home/shangguanrz/project/pic-edit/experiments/qwen21_dual_v8/eval_pair14_v8_300_v1 \
  --output /home/shangguanrz/project/pic-edit/experiments/qwen21_dual_v8/rescore_reproduce_v1 \
  --steps 40 --seeds 42 43 44 --cfg 1 \
  --mesh-metrics --mesh-python "$MHMR_PY" --mesh-device cuda
```

输出含 `original_run.json`、`original_probes.json`、新 `run.json` 与逐例评分 JSON、`multihmr_raw/request.json`、raw NPZ/sidecar、`results.json`。本轮原始 v1 因 Qwen 环境缺 `roma` 而接触 NA；v2 使用现有独立环境后排除了环境失败，但 V8 seeds43/44 仍因重建只有1人而 NA，不能把退出码0当作全指标通过。

## 结果如何理解

- DWPose 身体 PCK 使用生成 RGB 检测结果对照外部 pose 控制的可靠点，衡量控制遵循；wholebody 对照实际相机观测。距离按512画布对角线归一化，0.05约36.2像素；拟合先验 confidence=0.25 被排除，缺人/遮挡单列失败或 NA。
- DINO 是外观/衣服代理，不能替代人脸身份；ArcFace 无可用脸时为 NA。
- 接触距离来自生成 RGB 的 Multi-HMR 重建和 A/B 几何匹配；SMPL 原接触到 SMPL-X 的转移及重建均是代理。不能用输入控制网格的距离冒充生成成绩。
- 读取接触均值时同时检查边分母和 `mesh_failed_outputs`；未重建两人的整图失败不会形成可计分的边，不能读成零距离或成功。
- 不足两人、第三人、无法分配的人和缺失指标都会单列；不能只报幸存样本均值。
- 本轮不验证双人精细脸/手，也未获得完整训练好的旧 full single 检查点来做 iPER 图像数值回归。已验证原 single 源码前缀不变、旧类与 dispatch 兼容；两者不等于完成生产图像回归。

质量结论以最终报告及图板为准；训练 loss 下降或模型运行成功本身不作为效果达标。
