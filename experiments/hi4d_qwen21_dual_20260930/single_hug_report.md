# Qwen-Image-2.1 双人单样本测试（Hi4D pair01/hug01）

## 样本与训练

- 同框源图：`pair01/cam4/hug01/000016`，两人尚未接触。
- 目标：`pair01/hug01/cam4/000087`，拥抱峰值；Hi4D 标注接触顶点共 3047 个，控制缓存含 8 条有效接触关系。
- 从 iPER 20k body 检查点初始化单人参数，双人新模块随机初始化。只在这一条训练样本上更新 300 步：前 100 步冻结原单人部分，后 200 步联合训练；无 iPER 重放，批量 1。
- 固定噪声下的同样本 flow 验证损失：100 步 0.5679，200 步 0.5336，300 步 0.5215。这是训练样本重建测试，不代表留出集泛化。
- 最佳检查点：`experiments/qwen21_hi4d_dual/single_hug_overfit/checkpoint_best.pt`（300 步）。

## 40 步采样结果

每个种子分别用相同噪声比较原生 Qwen（off）、仅几何（geometry）、完整双人（full）。

| 种子 | 模式 | 可见结果 | A 姿态 PCK@0.05 | 检出人数 | 背景 MAE |
| --- | --- | --- | ---: | ---: | ---: |
| 42 | off | 面对面握手 | 0.077 | 2 | 6.37 |
| 42 | geometry | 两人靠近，未形成完整拥抱 | 0.308 | 1 | 5.87 |
| 42 | full | 接近目标的双人拥抱，衣着和场景保持较好 | 0.308 | 1 | 4.06 |
| 43 | off | 两人仍分开 | 0.000 | 2 | 8.74 |
| 43 | geometry | 出现第三个人物 | 0.385 | 3 | 9.62 |
| 43 | full | 形成拥抱，但背景明显位移 | 0.308 | 1 | 49.48 |

两张 full 图中的蓝上衣/红短裤与黑衣人物仍对应源图的 A/B，肉眼未见衣着互换。
自动 DWPose 检测器对拥抱图只检出 1 人，因此 B 的姿态 PCK、接触距离和
自动身份交换率均不可可靠计算，评测文件记为 `null`。仅几何 seed 43 检出了
3 人，自动身份交换率同样记为 `null`。这是评测可观测性限制，不能用缺失值
证明双人指标达标。

图板：

- `experiments/qwen21_hi4d_dual/single_hug_eval/images/0000_panel.png`
- `experiments/qwen21_hi4d_dual/single_hug_eval_seed43/images/0000_panel.png`

原始指标：各目录的 `summary.json` 和 `per_sample.json`。

## 复现

在 `ssh sgrzhh` 上：

```bash
cd /home/shangguanrz/project/pic-edit
PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
grep -m 1 'target_is_episode_peak.*true' datasets/Hi4D_pilot_v1/final_assets_v4_2/manifests/main_pairs_train.jsonl > /tmp/qwen21_hi4d_single_hug_train.jsonl
$PY train_hi4d_qwen21_dual.py --train-pairs /tmp/qwen21_hi4d_single_hug_train.jsonl --val-pairs /tmp/qwen21_hi4d_single_hug_train.jsonl --output experiments/qwen21_hi4d_dual/single_hug_overfit --max-steps 300 --stage1-steps 100 --iper-replay-rate 0 --micro-batch-size 1 --accumulation 1 --val-count 1 --save-every 100
$PY eval_hi4d_qwen21_dual.py --checkpoint experiments/qwen21_hi4d_dual/single_hug_overfit/checkpoint_best.pt --pairs /tmp/qwen21_hi4d_single_hug_train.jsonl --output experiments/qwen21_hi4d_dual/single_hug_eval --max-samples 1 --steps 40 --seed 42 --allow-val
$PY eval_hi4d_qwen21_dual.py --checkpoint experiments/qwen21_hi4d_dual/single_hug_overfit/checkpoint_best.pt --pairs /tmp/qwen21_hi4d_single_hug_train.jsonl --output experiments/qwen21_hi4d_dual/single_hug_eval_seed43 --max-samples 1 --steps 40 --seed 43 --allow-val
```

已有图像需要重新计算指标时，给对应评测命令加 `--reuse-images`，不会重新加载 Qwen 或生成图像。

## 结论

双人控制在这条训练样本上能把非接触源图引导为拥抱，证明前向、反向、A/B
外观绑定和接触条件链路可以工作。两个种子的背景稳定性不同；仅几何分支
曾生成第三人。需要继续处理多人物重复与背景漂移，并用对拥抱遮挡更稳健的
评测方法确认 B 姿态和接触，之后才能声称双人效果达标。全量训练目前暂停。
