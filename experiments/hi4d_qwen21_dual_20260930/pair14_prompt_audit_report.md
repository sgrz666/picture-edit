# Hi4D pair14 双人动作提示词与数据边界排查

## 范围

- 样本：`pair14`，同一张双人源图 `talk14/cam4/frame32`，目标动作 `hug14/cam4/frame62`，512×512。
- 检查点：旧通用双人提示词的 300 步单样本过拟合检查点；其训练样本是 `pair01`，没有训练本条 `pair14`。本报告只排查提示词、控制敏感性和目标真值泄漏，不作为泛化或质量验收。
- 每组使用相同源图、40 步、种子 42、43、44。原生 Qwen 只接收源图和文本；`geometry_short` 接收目标姿态、部位、法线、深度、相对几何与渲染人物区域，不接收接触；`full_short` 额外接收目标接触及 A/B 外观绑定。后两组是**目标几何已知的控制任务**。渲染人物区域来自 SMPL/geometry 数据，不是评测真值人物掩码文件。

原生详细提示词：

> Edit the supplied photo of exactly two existing people. The person on the left and the person on the right move toward each other and hug. Their torsos touch and their arms wrap around each other's shoulders and backs. Preserve each person's face, hair, clothing and proportions, as well as the camera view, lighting and background. Do not add another person or change the setting.

Adapter 简短提示词：

> Make the two existing people hug. Preserve their appearance and background.

提示词由动作标签及源图 A/B 掩码的左右关系确定。训练与推理共用 `hi4d_dual_action_v2` 模板。

## 输入边界与控制切换

- 评测数据集以 `load_target=False` 取样；目标 RGB 在所有生成完成后才为图板和指标读取。生成入口只接受源图、提示词和白名单控制张量。
- 在内存中替换目标 RGB、两张真值人物掩码及目标路径后，同一控制和种子 42 的 `full_short` 输出**逐像素相同**，最大绝对差为 **0**。
- 固定源图、提示词和种子 42，将目标控制从 `hug14/frame62` 切到 `hug14/frame34`，控制图的姿态、部位、法线、深度、接触等字段改变，输出平均绝对像素差 **4.670/255**。两张输出的手臂、头部和脚位置可见差别。

## 五组结果

PCK 为归一化阈值 0.05。`—` 表示检测不足而无法评估，绝不按 0 或成功计入。背景误差是生成图与**源图**的非人物区域 MAE，越低越好。第三人检出三种种子均为 0。目标真值图在相同 512 分辨率下也仅检出 1 人，拥抱遮挡使 B 姿态与接触指标不稳定。

| 种子 | 组别 | 目标几何 | 检出人数 | A PCK | B PCK | 接触距离误差 | 背景 MAE |
|---:|---|---|---:|---:|---:|---:|---:|
| 42 | 原生＋详细 | 无 | 2 | .33 | .25 | .113 | 4.43 |
| 42 | 仅几何＋简短 | 有，无接触 | 2 | .47 | .31 | .111 | 4.53 |
| 42 | 完整双人＋简短 | 有，含接触 | 1 | .27 | — | — | 3.07 |
| 43 | 原生＋详细 | 无 | 1 | .33 | — | — | 3.72 |
| 43 | 仅几何＋简短 | 有，无接触 | 1 | .40 | — | — | 3.61 |
| 43 | 完整双人＋简短 | 有，含接触 | 1 | .20 | — | — | 3.11 |
| 44 | 原生＋详细 | 无 | 2 | .33 | .31 | .075 | 3.90 |
| 44 | 仅几何＋简短 | 有，无接触 | 1 | .20 | — | — | 4.31 |
| 44 | 完整双人＋简短 | 有，含接触 | 1 | .27 | — | — | 3.14 |

另有 `off_short` 和 `full_detailed` 两组交叉提示词对照，逐例指标及图像见各种子目录。种子 42 的目标 RGB MAE：原生详细 9.02、仅几何简短 7.91、完整双人简短 5.63；源图对目标为 12.43。完整组看起来接近目标，但**并非逐像素相同**，且拥有目标帧导出的密集几何。这个差值不能归因于提示词或泛化。

## 结论与下一步

本次没有发现目标 RGB 或评测真值掩码穿过生成入口；目标几何切换会改变输出。旧 300 步检查点在此条未见训练的配对上不能证明双人姿态或接触优于原生模型，尤其 B 与接触指标大多无法评估。下一步使用新动作提示词从 iPER 单人检查点训练多样本双人模块；只用验证集选检查点，再在测试集做正式比较。

## 复现

在 `/home/shangguanrz/project/pic-edit`，使用 `/home/shangguanrz/nihaochen/envs/qwen21/bin/python`：

```bash
python eval_hi4d_qwen21_dual.py \
  --checkpoint experiments/qwen21_hi4d_dual/single_hug_overfit/checkpoint_best.pt \
  --pairs /tmp/qwen21_hi4d_pair14_peak_audit.jsonl \
  --output experiments/qwen21_hi4d_dual/prompt_audit_pair14_v2/seed42 \
  --max-samples 1 --steps 40 --seed 42 --allow-val \
  --allow-legacy-checkpoint --prompt-matrix
python audit_hi4d_qwen21_prompt_leak.py \
  --peak-pairs /tmp/qwen21_hi4d_pair14_peak_audit.jsonl \
  --alternate-pairs /tmp/qwen21_hi4d_pair14_early_audit.jsonl \
  --checkpoint experiments/qwen21_hi4d_dual/single_hug_overfit/checkpoint_best.pt \
  --output experiments/qwen21_hi4d_dual/prompt_audit_pair14/leak_and_geometry \
  --steps 40 --seed 42
```

`seed43` 和 `seed44` 只需同步改 `--seed` 与 `--output`。已生成的图像可用 `--reuse-images` 重新打分；它会核对检查点、数据清单、提示词和采样参数。
