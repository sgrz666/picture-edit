# Qwen-Image-2.1 双人主体控制

入口位于远端 `/home/shangguanrz/project/pic-edit`。输入是一张同框的双人源图及
对应 A/B 源掩码，目标是 Hi4D 提供的渲染几何。训练和推理均在 512×512 等比
填充画布上运行。原有 iPER 单人入口不变。

## 数据契约

`src/data/hi4d_qwen21_dataset.py` 读取 `final_assets_v4_2` 主配对 manifest。
训练使用 `main_pairs_train.jsonl`，选择检查点只用 `main_pairs_val.jsonl`，
最终报告只用 `main_pairs_test.jsonl`。转换器将目标 SMPL 网格与相机投影为
14 类部位图，将 DWPose 变成 25 通道热图，并组织相对位置、接触图与最多八条
接触关系。源人物外观由同一源图的 A/B 掩码分别绑定。目标真值 RGB 仅用于
训练监督；`target_person_mask_A/B` 从未作为控制输入。

控制缓存版本为 `hi4d_qwen21_dual_v2`，模型/提示词缓存带有模型 revision、
提示词及图像字节哈希。更新画布或控制定义时必须更新控制缓存版本。

## 训练

```bash
cd /home/shangguanrz/project/pic-edit
PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
$PY train_hi4d_qwen21_dual.py --precompute-controls-only
$PY train_hi4d_qwen21_dual.py
```

训练默认从 iPER 20k 最佳 body 检查点继承单人路径；前 2000 步冻结该路径，
后 8000 步以较低学习率联合训练并混入 10% iPER 单人重放。
有效批量为 4（每次前向 2 条、累积 2 次）。每 500 步在固定 64 条验证子集上选择检查点，
训练结束再以最佳检查点评测全部 822 条验证配对，并保存
`experiments/qwen21_hi4d_dual/checkpoint_best.pt` 与
`checkpoint_last.pt`；全量验证结果写入 `validation_full.json`。
中断后用 `--resume` 指向后者，保持原配置。

## 推理及评测

```bash
$PY eval_hi4d_qwen21_dual.py --checkpoint experiments/qwen21_hi4d_dual/checkpoint_best.pt
```

默认对留出的 606 条测试配对生成 native、仅几何、完整双人三组图像，写入
`experiments/qwen21_hi4d_dual/test_eval`。单例查看可加 `--max-samples 1`；
验证集开发调试须显式加 `--pairs .../main_pairs_val.jsonl --allow-val`。
评测报告包括 A/B 姿态 PCK@0.05、接触部位距离误差、背景 MAE、
VGG16 身体外观相似度及身份交换率。身体外观相似度是衣着/外观代理指标，
不是人脸识别分数；不可检测的姿态或接触项会记录缺失及样本数。

## 验证

```bash
$PY -m pytest -q tests/test_qwen21_dual_control.py tests/test_qwen21_control_contract.py
$PY -m py_compile src/data/hi4d_qwen21_dataset.py src/pose_control/qwen21/dual_adapter.py train_hi4d_qwen21_dual.py eval_hi4d_qwen21_dual.py
```

端到端冒烟可从 train/val manifest 各取一行放在 `/tmp`，使用 `--max-steps 2
--stage1-steps 1 --accumulation 1 --val-count 1 --save-every 2` 训练，再用
评测入口 `--max-samples 1 --steps 4 --allow-val` 生成三组图。
