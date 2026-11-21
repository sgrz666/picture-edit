# Native V7 双人单配对过拟合

## 实验目的

验证真实 Qwen Image 2.1 + Native V7 能否学习一个固定双人姿态配对，同时保留两人的身份。训练只使用 `pair12/highfive12/cam4` 的 Source 000011 → Target 000082；同人物 Target 000112 仅观察，不参与优化，不作为泛化测试集。

首轮上限 300 个 optimizer updates，分为 Body 30、Head 30、PoseLink Full 120、Joint 120；必须逐阶段通过数据、真实梯度和冻结检查。固定 seed 42、原生 1280×960 Canvas、4Ref、B1。用户随后明确选择继续原生尺寸并已暂停 ComfyUI，新 run 显式使用 90% allocator 上限，保留完整 ID 监督；旧的 50% 预算失败记录保留。

## 里面有什么

- `scripts/audit_inputs.py`：核对 typed 样本、生产初始化权重和冻结 evaluator。
- `scripts/prepare_assets.py`：匹配真实检测的人脸与稳定人物槽位，复用 `training_sidecar()`、冻结 evaluator 工厂和原始 Source Mask 生成新侧车。
- `scripts/configure_run.py`：生成 SHA 绑定的 1 训练行、1 观察行与 30/30/120/120 阶段配置。Head 阶段同时预热 Face/Hand/Interaction/PoseLink，避免 Joint 门禁碰到零头。
- `scripts/gpu_action.py`：共享 Trainer 的实验包装，默认 50% 显存上限，可显式传入 `--memory-fraction 0.9`；首次 optimizer update 显存测量和执行策略进入严格 Backend identity。可选官方 VAE tiling 或精确 Hidden CPU 暂存，新 run 两者均关闭。
- `scripts/run_training.py`／`launch.py`：逐阶段真实门禁、保存、技术 Body 检查、生成与 PID/状态；不自建训练循环。
- `scripts/status.py`：只读输出当前 run 各阶段的最新真实更新数、梯度、ID 更新次数和失败／完成报告，省去打印完整逐参数诊断。
- `scripts/rebind_paths.py`：数据别名失效后，只为本实验重绑到实存路径；每个引用必须匹配原 SHA256，重新验证 grounded 条件与缓存语义，不修改数据集或恢复全局软链。
- `scripts/audit_rebound.py`：只读核对原 SHA 与新旧缓存所有值逐项相等；本轮 `path-rebind-audit.json` 已通过，唯一允许改变的缓存字段为条件文件 SHA。
- `scripts/evaluate_run.py`：固定噪声 Joint Loss、SDPose 姿态及 CPU 身份评价；分别使用现有 qwen21／SDPose 环境，每次必须使用新的报告目录。
- `scripts/audit_pose_control.py`：真实 Target GT 的 CPU 姿态正对照，每样本一次全图检测；复用同一套共同 12 关节、投影和身份槽位匹配。
- 服务器实验根目录：`/data/shangguanrz/pic-edit-experiments/qwen21_hi4d_dual/native-v7-singlepair-overfit-20261117-v1/`。新增合格监督与 repair 在 `data-v2/`；四图缓存在 `cache/native-4ref/`；生产迁移在 `initialization/`；所有实际日志、失败和资源峰值在 `logs/`、`report/`，不会覆盖旧 pilot。

## 检查标准与当前状态

1. 两个人的 GT 五点必须来自真实检测，并与稳定人物槽位及 Source 身份核对；不可见人脸不填伪标签。
2. repair 只使用源人物 Mask 和已核对的 Source→Target Canvas 变换，不能从 Target RGB 反推。
3. 从未对本实验目标配对训练的生产初始化迁移到 V7，不用旧 Hi4D 300 步结果或 CPU 合成权重冒充新实验。
4. 看固定噪声 FM、逐人物 ID/对比项、SDPose 姿态与可视化生成；记录训练与观察样本的差异。Loss 下降不单独证明成功，观察样本改善也不证明跨身份泛化。

真实数据资格、生产迁移和四图缓存均已完成，当前完整通过 **S0 30 步、S1 30 步**，S2 真实梯度预检也已通过并开始训练；尚不能宣称已经过拟合。先前 50% 显存预算的尝试均为 0 次更新，失败记录如下：

|预检方式|实测结果|
|---|---|
|原生 VAE 解码|在 41.52GiB allocator 上限触发 OOM；保留 ID Loss，没有优化更新|
|官方 VAE tiling|反向图仍占内存，仍触发 OOM；没有降低 Loss 或伪造门禁|
|大 Hidden 暂存 CPU，原生 VAE|暂存 34 个 Tensor、7,028,080,640 字节，仍在 VAE 卷积处触发 OOM|

训练帧两人的 Source–Target ArcFace 余弦为 **0.8364／0.8285**，观察帧为 **0.8656／0.8081**，交叉人物余弦为负；四个人脸实例都通过真实五点检测、唯一人物匹配及 ONNX/TorchScript ≥0.999 一致性检查。两条样本的辅助资格均为 `[true,true]`，repair 只用源 Mask。CPU数据准备回归与真实 CUDA Hidden 暂存输出/梯度逐值一致性检查通过。

用户已暂停原先占约 35GiB 的 ComfyUI。`pilot300-native-fullgpu-v2` 使用原生尺寸、90% 上限、不开 tiling／Hidden 暂存；S0 真实梯度预检通过，用时 **304.8 秒**，peak allocated **58,168,345,600 字节（54.17GiB）**、reserved **66,985,132,032 字节（62.38GiB）**。程序随后生成未训练对照，再进入真实 optimizer 更新；是否完成以 `completed.json`／阶段保存为准。

本轮 S0 完整通过 **30 个真实 optimizer updates**，4 次更新对两个人脸执行 ID／contrastive；阶段用时 **1537.8 秒**。S0 peak allocated **58,555,066,368 字节（54.53GiB）**、reserved **68,371,349,504 字节（63.68GiB）**。提交的阶段 checkpoint 为 `update_00000030_782f5d3a6bc140de8b1396fde54977dc.pt`，SHA256 `112c7209beabe7591ac96880c810a3e9e274cd84e8174aab76df141b3fd1b2cd`。首次更新单独峰值不代表整个阶段峰值。

训练前四张 20-step／seed42 对照均已保存，原图 940×1280，仅右侧补齐到 Canvas 960×1280（宽×高），未缩放。可见旧手臂残留与击掌不齐。CPU SDPose 全部检出两人、24/24 个共同关节有效；正式基线 `baseline-sdpose-cpu-seeded-v2/evaluation.json` 如下，PCK 阈值以各人物 GT torso 长度归一化：

| 样本／配置 | PCK@0.1 | PCK@0.2 | NME | 关节误差 px | 背景 MAE |
|---|---:|---:|---:|---:|---:|
|训练／Qwen only|54.17%|66.67%|0.23365|52.35|4.8349|
|训练／初始 Native|54.17%|62.50%|0.23269|52.18|4.8764|
|观察／Qwen only|29.17%|58.33%|0.23411|47.30|4.6459|
|观察／初始 Native|29.17%|58.33%|0.26940|53.62|4.5734|

GT 姿态正对照在 `pose-gt-control-cpu-seeded-v2/evaluation.json`：训练／观察均检出 2 人，分别 24／23 个共同点有效，PCK@0.2 均为 **100%**，关节误差 **12.06／11.71px**。观察样本有一个 right_elbow 未通过检测有效条件，保留该缺失。该约 12px 误差包含几何拟合、遮挡与估计器差异，不能从生成误差中直接相减。

SDPose 的 VAE 内部有 `latent_dist.sample()`；正式评价每次全图／区域检测都独立固定 seed **24042**，用 `fork_rng` 恢复调用方状态，只处理 CPU 和当前 runtime 的 CUDA 设备。原 `*-cpu-v1` 未逐调用锁 seed，保留为探索记录，不作为最终比较基线。新报告绑定实际 pipeline SHA `fa39f7f6e53547f5b14ed31458a6dbf47f812afd2ae89218473b741c6da48d20`；本次复测的 PCK 与 v1 相同，浮点末位略有变化。

S1 四路预检通过，用时 **561.1 秒**；S1 的 30 步也已完成（**1434.8 秒**）。累计 60 步 checkpoint 为 `update_00000060_5042c056c9b64d08b0d34d3b16313df6.pt`，SHA256 `09b19daa36e4f429db2f2aa4c5ecde78dfe9493dddd53f5583c87438f2c5d8a0`，保留 optimizer 父链。S2 预检通过，用时 **260.3 秒**，peak allocated **51,989,270,016 字节**，已完成首个真实更新。全部阶段是否完成以各自 commit／summary 和最终 `completed.json` 为准。

`identity-positive-control-v1/evaluation.json` 验证真实 GT 的对齐路径：训练两人 Source–GT 为 **0.83778／0.83196**，观察两人为 **0.86517／0.80939**，相对准备阶段最大差 **0.00347**；人物槽位与 Canvas 一致。生成图的分数同时受 GT 位置对齐误差与身份还原影响，需要结合图像判断。

当前运行源码备份：`/data/shangguanrz/code-snapshots/native-v7-pilot-runtime-20261121T0330Z/runtime.tar.gz`，SHA256 `9fa97a5087d18b34d009856072391a7d8a57a1d8a69c4b05d91dcd33a5abe9e5`。此文件保存服务器实际源码；服务器 Git HEAD 仍为 `00fb054d`，使用已审计源码覆盖层，不将该旧 HEAD 冒充 V7 实际执行版本。

新 run 读取 `data-rebound-v1/`、`cache/rebound-v1/` 与 `configs/native-fullgpu-paths-v1/`。原先的 `/data/shangguanrz/pic-edit-datasets` 别名失效，实际图像仍在 `pic-edit-datasets-test`；所有重绑引用均验证原 SHA，Prompt、Canvas、图像角色及缓存 Tensor 不变。早先 `pilot300-native-fullgpu-v1` 因旧 Source 路径失效在 0 更新退出，记录保留。

Hidden 暂存使用 [PyTorch saved tensor hooks](https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html)，只移动大的三维 4096 通道激活，排除权重和 Mask。对应策略、脚本 SHA、VAE 实现 SHA 都进入真实 Backend identity；没有修改图像、模型参数或损失计算精度。

复核数据（`--report` 必须是新的文件）：

```bash
cd /home/shangguanrz/project/pic-edit
PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
RUN=/data/shangguanrz/pic-edit-experiments/qwen21_hi4d_dual/native-v7-singlepair-overfit-20261117-v1
$PY scripts/train/train_qwen21_native_full_v7.py preflight --level data \
  --config "$RUN/configs/native-fullgpu-paths-v1/S0.json" --report "$RUN/report/data-check-new.json"
tail -n 5 "$RUN/logs/pilot300-native-fullgpu-v2-events.jsonl"
$PY "$RUN/scripts/status.py" --root "$RUN" --name pilot300-native-fullgpu-v2
```

新建另一轮（仅在现有训练结束且 GPU 足够空闲时，`--name` 必须未使用）：

```bash
$PY "$RUN/scripts/launch.py" --root "$RUN" \
  --templates "$RUN/configs/native-fullgpu-paths-v1" \
  --name pilot300-native-new --memory-fraction 0.9
```

查看当前结果：`output/pilot300-native-fullgpu-v2/baseline/` 与 `final/`；四阶段 checkpoint 位于 `output/training/`，准确 SHA、累计更新数及父链见 `report/pilot300-native-fullgpu-v2/`。最终生成尚未完成时，`final/` 不存在属于预期状态。

300 步和最终生成完成后顺序评价（新的 `report-dir` 不得已存在，GPU 评价与训练错开）。下列命令只评最终两张图，与已保存的 seeded-v2 基线比较，避免重复检测：

```bash
NAME=pilot300-native-fullgpu-v2
EVAL="$RUN/scripts/evaluate_run.py"
CFG="$RUN/configs/$NAME/final-generation.json"
FINAL="$RUN/report/$NAME/final-generation.json"
$PY "$EVAL" --action fixed-probe --config "$CFG" \
  --initial-checkpoint "$RUN/initialization/v7.pt" \
  --report-dir "$RUN/report/$NAME/eval-fixed-v1"
env CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  CUDA_HOME=/home/shangguanrz/nihaochen/envs/qwen21 \
  /data/shangguanrz/pic-edit-experiments/four-skeleton-compair/.venv/bin/python \
  "$EVAL" --action image-metrics --pose-device cpu --config "$CFG" \
  --generation-report "$FINAL" \
  --report-dir "$RUN/report/$NAME/eval-sdpose-v1"
$PY "$EVAL" --action identity-metrics --config "$CFG" \
  --generation-report "$FINAL" \
  --prepared "$RUN/data-v2/prepared.json" \
  --identity-model /home/shangguanrz/project/pic-edit/models/face_eval/identity_encoder.ts \
  --report-dir "$RUN/report/$NAME/eval-identity-v1"
```

GT 姿态控制可单独复测：在同一 SDPose CPU 环境运行 `audit_pose_control.py --config "$CFG" --evaluator "$EVAL" --report-dir <新的绝对目录>`。这只读取真实 GT，不把 Target RGB 用作生成条件。

固定噪声比较使用相同 Joint 目标、sigma 0.2／0.5／0.8、每样本 noise seed 24042。姿态由 SDPose-Wholebody + YOLO11x 提取，score≥0.30，`order=geometry`，只比较 Body21／Body18 对应的 12 个身体关节；未检测到两人会显式记录，不能从均值里丢掉。身份指标是在目标五点位置对齐后的 Source→Generated 2×2 余弦矩阵，位置不准也会影响分数，需要结合姿态和图像解释。

私有脚本回归：`python -m pytest tests/test_native_v7_singlepair_preparation.py tests/test_native_v7_singlepair_driver.py tests/test_native_v7_singlepair_eval.py -q`，**52 passed、1 skipped**。跳过项为本地 CPU-only 机器上的真实 CUDA 暂存测试，之前已在服务器核验输出/梯度逐值一致。服务器可用 `V7_PILOT_SCRIPT_ROOT` 指定实验脚本路径后复测。

## 与其他实验的关系

上游为 Native V7 架构（`1838e58`）和 Hi4D r4 grounded binding pilot 的两条 typed 4Ref 数据。本实验补齐训练资格与修复条件，沿用共享 Trainer；后续少样本、多人物组合训练以本实验的真实可学性与效果为依据。
