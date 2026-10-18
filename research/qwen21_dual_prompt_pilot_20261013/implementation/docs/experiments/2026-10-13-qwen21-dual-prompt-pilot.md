# 双人逐样本动作提示词试验：运行与检查

2026-10-14更新：真实数据准备、short/structured各300步训练和单条独立测试配对的12张生成/评分均已完成。两支最终 best 均为验证选择的第50步（short 0.1123397934、structured 0.1124632967）。测试 guard 核验无反向传播、无 optimizer 操作、模型/adapter权重前后不变。实际结果、指标分母和局限见 `reports/Qwen21_dual_prompt_pilot_20261013.md`：本次没有证实结构化文本稳定改善 adapter。

原SSH前台训练在 structured 部分中断；严格续训在梯度范数不一致时停止，最终保留已完成 short300，从相同初始化完整重训 structured300。没有将未完成或重放失败的分支作为正式结果。根完成记录明确包含 recovery_provenance；原中断进程最终内存 SHA 未捕获，新 structured 和测试进程的前后权重 SHA 已实测。复现可能受 GPU 浮点规约影响，不能保证逐位相同的曲线或图像。

本轮比较 `short` 与 `structured` 提示词：各自从相同 iPER body step19500 和相同新 adapter 初始化独立训练，使用相同样本顺序、噪声、损失、300 optimizer steps、有效 batch4、lr1e-4。每 50 步在真正的验证配对上使用固定种子 420001/420002/420003 计算无梯度 flow loss，据此选择 best。测试读取已冻结的 best，不进行生成模型优化。

## 数据与提示词

| 划分 | pair | 源动作/帧 | 目标动作/帧 | camera |
|---|---|---|---|---|
| train | pair14 | talk14/32 | hug14/45 | 4 |
| train | pair14 | talk14/32 | hug14/62 | 4 |
| train | pair12 | highfive12/28 | highfive12/33 | 4 |
| train | pair12 | highfive12/28 | highfive12/82 | 4 |
| val | pair22 | game22/52 | hug22/46 | 4 |
| test | pair00 | taichi00/41 | hug00/108 | 4 |

这是四条训练配对、一个验证配对、一个测试配对；train/val/test 的 pair 互斥。数据没有跨 pair 的全局演员身份验证，因此不宣称全局演员身份互斥。940×1280 源图通过等比缩放和填充进入 512×512；RGB 和掩码使用同一仿射变换。

真实 SMPL-X 拟合与数据导出已执行：每帧 200 个离线 mesh fitting steps，冻结 SMPL-X 模型，仅优化网格姿态、根部、平移及源帧形状参数；目标固定源 actor 的 beta10。拟合过程没有载入 Qwen 或 adapter，也没有用目标 RGB/真值掩码约束几何。数据 JSON 记录的准备耗时为 118.11 秒，整个准备进程墙钟约 127 秒。输出为 SMPL-X 的 10,475 顶点/20,908 面，身体关节拟合均值约 10–13 mm、最大单关节误差小于 38 mm；这是拟合代理，不是官方 SMPL-X 真值。

逐关节置信度是拟合误差的启发式值，未校准。手指采用未观测的模型先验，手部置信度上限为 0.25；脸部 jaw/expression/eye 也采用先验。提示词不会把这些先验写成确定的手指或表情动作。

精确文本保存在 `dataset.json` 的 `selected_samples[].prompts.short/structured`，来源审计在 `prompts.provenance`，模板为 `source_geometry_action_v1`。结构化文本仅依据源掩码人物指认、动作标签和可靠的外部目标几何，未通过目标 RGB 编写。A/B 始终绑定输入 UID 的 actor0/actor1，并按源图位置指认；测试配对的 **A 在源图右侧、B 在左侧**。不能根据目标位置重新分配人物。

两条 hug 训练帧的粗粒度提示词相同，这是阈值离散后得到的真实结果；两条 highfive 的低位与高位阶段具有不同几何描述。逐样本文本的要求是准确绑定样本和可靠动作，不要求为了唯一性加入虚构细节。

GT RGB/真值掩码在训练中只用于 FM 监督，在正式测试中仅于全部 12 张图生成完成后打开评分；生成接口不接收它们。独立外部目标几何是本轮任务约定的输入。原生 Qwen 的结构化文本同样包含从该几何提取的描述，评测分别记录 `explicit_target_geometry_control` 和 `prompt_geometry_descriptions`，不能将原生结构化组描述为“没有任何几何信息”。

## 位置与环境

| 项目 | 位置 |
|---|---|
| SSH | `ssh sg` |
| 原工程/模型 | `/home/shangguanrz/project/pic-edit` |
| 本轮隔离实现 | `/home/shangguanrz/project/pic-edit/research/qwen21_dual_prompt_pilot_20261013/implementation` |
| 本轮数据 | `/home/shangguanrz/project/pic-edit/research/qwen21_dual_prompt_pilot_20261013/data` |
| 已完成训练/测试 | `/home/shangguanrz/project/pic-edit/experiments/4.two man /qwen21_dual_prompt_pilot/20261013_v1` |
| Qwen Python | `/home/shangguanrz/nihaochen/envs/qwen21/bin/python` |
| Multi-HMR Python | `/home/shangguanrz/project/pic-edit/add/envs/multihmr/bin/python` |
| warm start | `/home/shangguanrz/project/pic-edit/experiments/3.singel man/qwen21_iper_body/full_20k/checkpoint_best.pt` |

`4.two man ` 的目录名末尾含一个空格；所有含空格的路径及变量展开必须加引号。warm start 的实际 step 为 19500，文件 SHA256 为 `9c2a274a2a10c94f4632f7f24f291868346975da71367dd7745d9390141b6a54`。没有使用上轮双人单样本 300 步 adapter 作为初始化。

复用本地 `codex/qwen21-dual-v8` 工作树；远端实现从先前完整镜像继承已有未提交基线，再加入本轮新文件。旧实验输出和无关改动保留，基线快照 ZIP 一并保留。复现应使用实际镜像/快照及 manifest 的源码签名，不能仅根据 Git HEAD 推断运行时全部代码。图表脚本仅读取结果，CPU 即可运行。

登录后设置：

```bash
cd '/home/shangguanrz/project/pic-edit/research/qwen21_dual_prompt_pilot_20261013/implementation'
export PYTHONPATH="$PWD:/home/shangguanrz/project/pic-edit"
PY='/home/shangguanrz/nihaochen/envs/qwen21/bin/python'
DATA='/home/shangguanrz/project/pic-edit/research/qwen21_dual_prompt_pilot_20261013/data'
RUN='/home/shangguanrz/project/pic-edit/experiments/4.two man /qwen21_dual_prompt_pilot/20261013_v1'
MHMR_PY='/home/shangguanrz/project/pic-edit/add/envs/multihmr/bin/python'
```

本轮使用单张 RTX 6000D、torch device0。NVML 初始化警告不阻止实际 CUDA 工作；是否执行成功以退出状态和产物检查为准。

## 代码回归测试

远端已执行新模块 83 项和历史 dual_v8 218 项，共 301 项通过。以下命令重跑这些有意义的边界检查，不启动正式 GPU 训练：

```bash
"$PY" -m pytest -q -c /dev/null --rootdir "$PWD" --confcutdir "$PWD" \
  tests/test_hi4d_dual_prompt_pilot.py \
  tests/test_qwen21_dual_prompt_pilot.py \
  tests/test_qwen21_dual_prompt_pilot_eval.py \
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

新增评测模块最初的 21 项包含对 `Tensor.backward`、`autograd.backward/grad`、`Optimizer.step` 和 optimizer 初始化的主动违规测试，以及状态恢复、实际张量 SHA 与不完整/错误训练记录拒绝检查。背景评分随后增加 2 项回归，23 项实际通过：根据数据审计中的原图尺寸和仿射变换排除 letterbox 黑边。上述 pytest 参数隔离祖先目录的配置，避免它把原工程的 `src` 插到隔离镜像前面。数据准备、训练和评测 CLI 的 `--help` 与编译检查已通过；这些代码检查不代替真实训练和真实生成证据。

## 复现训练

训练入口生产默认值即为两支 300 步、lr1e-4、每50步验证。指定新目录，现有目录会被拒绝：

```bash
TRAIN_REPRO='/home/shangguanrz/project/pic-edit/experiments/4.two man /qwen21_dual_prompt_pilot/20261013_reproduce_v1'
"$PY" -u scripts/train/train_qwen21_dual_prompt_pilot.py \
  --data-dir "$DATA" --output "$TRAIN_REPRO"
```

训练保存共享顺序/噪声 `shared_schedule.json`、缓存 SHA `cache_manifest.json`、根目录及两支 `run_manifest.json`、各支 `train_metrics.jsonl` 与 `checkpoint_best.pt/checkpoint_last.pt`。正式训练先在真实 Qwen 上检查新零输出 heads 与原生前向逐元素完全相同，再写入各支 `real_model_gates.json`。完成记录仅在两支结束并确认 Qwen 主干、warm start、缓存和源码不变后写入。

检查两支实际完成记录、验证选择和 best 文件（本轮已核验通过）：

```bash
"$PY" - "$RUN" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
read = lambda path: json.loads(path.read_text())
done = read(root / 'training_complete.json')
assert done['status'] == 'success' and done['production_training'] is True
assert done['completed_steps'] == done['requested_steps'] == 300
assert set(done['variants']) == {'short', 'structured'}
assert done['initialization_sha256_equal'] and done['backbone_unchanged']
assert done['backbone_sha256_before'] == done['backbone_sha256_after']
assert done['validation_selection'] == 'true_val_split_only'
assert done['test_data_read'] is False
for variant in ('short', 'structured'):
    branch = read(root / variant / 'training_complete.json')
    gate = read(root / variant / 'real_model_gates.json')
    assert branch['status'] == 'success' and branch['completed_steps'] == 300
    assert branch['validation_selection'] == 'true_val_split_only'
    assert gate['zero_init_exact'] and gate['gradients_none'] and gate['optimizer_steps'] == 0
    rows = [json.loads(line) for line in (root / variant / 'train_metrics.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rows] == list(range(1, 301))
    vals = [row for row in rows if 'validation_loss' in row]
    assert [row['step'] for row in vals] == list(range(50, 301, 50))
    best = min(vals, key=lambda row: row['validation_loss'])
    assert branch['best_step'] == best['step']
    assert branch['best_validation'] == best['validation_loss']
    checkpoint = root / variant / 'checkpoint_best.pt'
    sha = hashlib.sha256()
    with checkpoint.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(block)
    assert sha.hexdigest() == branch['best_checkpoint']['sha256']
    print(variant, 'best step', branch['best_step'], 'validation loss', branch['best_validation'])
print('Both training branches complete; best uses true validation only.')
PY
```

## 生成一条真实测试配对

评测预检查拒绝缺失、未完成、smoke、非 best 或 SHA/源码/数据不匹配的训练记录。通过后，四组 `native short/native structured/dual short/dual structured` 各用种子42/43/44，40步、CFG1，共12张图。每组使用同一源图；dual 接收显式外部几何控制，structured 文本带有可靠几何描述。

```bash
EVAL_REPRO='/home/shangguanrz/project/pic-edit/experiments/4.two man /qwen21_dual_prompt_pilot/20261013_eval_reproduce_v1'
"$PY" -u scripts/eval/eval_qwen21_dual_prompt_pilot.py \
  --data-dir "$DATA" --training-root "$RUN" --output "$EVAL_REPRO" \
  --steps 40 --seeds 42 43 44 --cfg 1 \
  --mesh-python "$MHMR_PY" --mesh-device cuda
```

整个生成阶段处于 `inference_mode`，全局 guard 禁止反向传播与 optimizer 操作，记录梯度开关、违规次数、参数 grad，以及原生/adapter 的实际权重前后 SHA。离线 test mesh fitting 优化的是独立网格参数，冻结的 Qwen 测试生成不训练 adapter；两者不可混淆。

现有结果位于 `$RUN/test_pair00_v1`。打开 `comparison_board.png` 看源图、GT 和12张输出；读取每张 PNG 的同名 JSON 及 `run.json` 查看逐人姿态、接触代理、身份/背景指标和覆盖率。`evaluation_complete.json` 必须有 `status=success`、`generated_outputs=12`；`run.json.target_supervision.opened_after_all_12_generations=true` 且 `never_passed_to_generate=true`。若存在 `evaluation_failed.json`，不能报告为正式评测完成。

以下检查仅读结果，不启动生成或训练：

```bash
"$PY" - "$RUN/test_pair00_v1" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
read = lambda name: json.loads((root / name).read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
done, run, guard = read('evaluation_complete.json'), read('run.json'), read('inference_guard.json')
assert done['status'] == 'success' and done['generated_outputs'] == 12
for key, name in [('run_sha256', 'run.json'), ('inference_guard_sha256', 'inference_guard.json'),
                  ('comparison_board_sha256', 'comparison_board.png')]:
    assert done[key] == sha(root / name)
assert guard['weights_unchanged'] and guard['all_parameter_grads_none'] and guard['all_models_frozen_eval']
assert guard['optimizer_instances_before'] == guard['optimizer_instances_after'] == 0
assert all(count == 0 for count in guard['blocked_attempts'].values())
assert len(guard['generation_samples']) == 12
assert all(not item['grad_enabled'] and item['inference_mode'] for item in guard['generation_samples'])
assert all(not item['grad_enabled'] and item['inference_mode'] for item in guard['native_forward_samples'])
assert guard['target_supervision_opened_during_generation'] is False
assert run['target_supervision']['opened_after_all_12_generations']
assert run['target_supervision']['never_passed_to_generate']
for record in run['output_records']:
    sidecar = root / pathlib.Path(record['path']).name
    assert sha(sidecar) == record['sha256']
    assert sha(sidecar.with_suffix('.png')) == record['png_sha256']
print('12 images verified; no test backpropagation or weight update.')
PY
```

漏检人物、缺失脸部或重建失败保留为 NA，同时展示成功/预期数量。本条真实 GT 图被相同 DWPose 评测器检出1人，图板中12张输出均可辨认两人；自动双人检出率不能直接解释为模型少生成人物。GT 检测仅用于评测器校准记录，未修改生成检测。DINO 是衣着/外观代理，DWPose 的人物框是空间代理，Multi-HMR 接触是估计相机坐标下的无符号邻近距离，不能替代穿透判定。一个 test pair 的三个种子不能建立总体泛化或双人质量门槛通过的结论。

## 绘制真实训练曲线与查看精确提示词

```bash
FIGURES='/home/shangguanrz/project/pic-edit/experiments/4.two man /qwen21_dual_prompt_pilot/20261013_figures_v1'
"$PY" scripts/reports/build_qwen21_dual_prompt_pilot_figures.py \
  --run-root "$RUN" --dataset "$DATA/dataset.json" --output "$FIGURES"
```

输出 `training_plot.png`、可编辑矢量 `training_plot.svg`、含日志 SHA 和实际点值的 `figure_sources.json`，以及从数据原文逐字导出的 `selected_prompts.md`。图左为四个 microbatch 的平均训练 flow 项，图右为真验证 flow loss；训练总目标包含额外 teacher/budget，另保存在来源 JSON 中。验证图仅画日志中的观测点，不连线插值，不做平滑。

脚本可在训练中运行：缺失的支和验证点不会补齐，最后一行未写完的 JSONL 会明确记为警告；损坏的完整行会报错。仅当根目录及两支生产完成记录与全部300步/六个验证点一致，才标注最终 best 星号。绘图检查范围为日志/完成记录一致性；checkpoint 的实际权重检查由上述命令和评测预检查负责。

也可把两支日志、根目录及两支完成 JSON 下载到本地保留相同目录结构，再传入本地 `--run-root`；`--dataset` 指向下载的真实 `dataset.json`。图表无需 GPU 或 Qwen 模型，Python 环境需已有 Matplotlib。每次生成指定新 `--output`，已有图表目录不会被覆盖。
