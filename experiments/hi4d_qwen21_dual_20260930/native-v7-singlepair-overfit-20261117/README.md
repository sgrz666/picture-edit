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
- `scripts/audit_detected_identity.py`：SCRFD 检测生成图真实五点，geometry 唯一匹配人物槽位，再用同一 frozen TS 比较 Source 身份；匹配失败保留 null，不把脸部位置改善当作纯身份改善。
- 服务器实验根目录：`/data/shangguanrz/pic-edit-experiments/qwen21_hi4d_dual/native-v7-singlepair-overfit-20261117-v1/`。新增合格监督与 repair 在 `data-v2/`；四图缓存在 `cache/native-4ref/`；生产迁移在 `initialization/`；所有实际日志、失败和资源峰值在 `logs/`、`report/`，不会覆盖旧 pilot。

## 检查标准与当前状态

1. 两个人的 GT 五点必须来自真实检测，并与稳定人物槽位及 Source 身份核对；不可见人脸不填伪标签。
2. repair 只使用源人物 Mask 和已核对的 Source→Target Canvas 变换，不能从 Target RGB 反推。
3. 从未对本实验目标配对训练的生产初始化迁移到 V7，不用旧 Hi4D 300 步结果或 CPU 合成权重冒充新实验。
4. 看固定噪声 FM、逐人物 ID/对比项、SDPose 姿态与可视化生成；记录训练与观察样本的差异。Loss 下降不单独证明成功，观察样本改善也不证明跨身份泛化。

**已完成原生尺寸的 300 个真实 optimizer updates、四阶段梯度／冻结／父链检查、最终生成和评价。** 本轮证明姿态可学：训练帧的旧手臂残影消失，观察帧也跟随另一目标姿态；但检测对齐身份分数与背景保真均有退步，暂不建议直接扩大训练。它仍是 1 个训练配对、1 个同人物观察配对，不能据此宣称跨人物泛化或证明每个模块的独立收益。

同一 SDPose CPU／seed24042 口径的最终比较如下，两张最终图都检出 2 人、24/24 个共同身体关节有效，髋心分配无歧义。PCK 阈值以各人物 GT torso 长度归一化。

|样本|PCK@0.1 初始→300步|PCK@0.2 初始→300步|NME 初始→300步|关节误差 px 初始→300步|
|---|---:|---:|---:|---:|
|训练|54.17%→87.50%|62.50%→95.83%|0.23269→0.06574|52.18→14.37|
|观察|29.17%→66.67%|58.33%→95.83%|0.26940→0.08891|53.62→17.26|

最终逐人物 PCK@0.2 均为 `[91.67%,100%]`；训练逐人物误差 `[14.38,14.35]px`，观察为 `[21.07,13.45]px`。观察帧左侧人物仍有更明显的姿态误差，手指、掌心接触质量需结合原图看，身体 12 关节分数不能替代手部评价。

固定噪声 Joint 比较采用相同 sigma 0.2／0.5／0.8 和每样本 noise seed24042，比较三点均值。训练联合目标 **0.545022→0.367761（-32.52%）**、Person FM **0.295914→0.248902（-15.89%）**、Contact FM **0.322000→0.047985（-85.10%）**。观察联合目标 **0.383772→0.379625（-1.08%）**，观察 Person FM **0.299540→0.302947**；观察 Contact FM 为0是没有有效接触区域，不表示接触预测完美。

背景并未完全保持：Source／Target 人物 Mask 并集之外，训练 MAE **4.8764→5.0002**、PSNR **29.820→27.140dB**；观察 MAE **4.5734→4.8142**、PSNR **30.966→27.204dB**，有效像素数量一致。后续训练要检查背景约束，不能只看姿态提升。

Target 五点位置对齐的 Source 身份余弦，训练 `[0.00405,0.75092]→[0.56371,0.75246]`，观察 `[-0.05113,0.72693]→[0.58685,0.76188]`。这不能解读为纯身份改善：改用生成图自身 SCRFD 真实五点对齐后，训练分数为 **`[0.71137,0.87347]→[0.56573,0.76999]`**，两人均下降，未触发身份交换标记。固定 GT 位置裁切分数的大幅提高主要包含脸部 placement 改善。

观察初始 Native 虽检出2张脸，但未通过严格 geometry 双人唯一匹配，检测对齐分数记 null；最终可测分数为 `[0.60652,0.83373]`，不能给出观察帧的纯身份 before／after 结论。真实 GT 的检测对齐正对照为训练 `[0.83778,0.83471]`、观察 `[0.86517,0.81230]`。检测对齐仍受姿态、可见性和检测质量影响，报告中保留了全部检测、五点与拒绝原因。

补充180步同seed／20-step／相同Joint推理控制的身份审计：训练初始→180→300为 `[0.71137,0.87347]→[0.57238,0.70942]→[0.56573,0.76999]`，观察180→300为 `[0.65071,0.87412]→[0.60652,0.83373]`。训练身份下降在Joint前已经存在，第二人在Joint后还有回升，不能把问题全部归咎于Joint。对应报告为 `control-update180-v1-generation.json` 和 `eval-update180-detected-identity-v1/evaluation.json`；此控制只增加推理与CPU评价，没有新增训练，推理显存上限显式为50%。

180→300步的姿态对照：训练 PCK@0.2 **91.67%→95.83%**、误差 **16.68→14.37px**；观察 PCK@0.2 **95.83%→95.83%**、误差 **17.271→17.261px**，均检出两人、24个共同点有效。因此额外Joint主要提高训练帧，观察姿态收益很小；观察身份分数略降。背景180步时已退步，180→300的MAE略降但PSNR略降，不能解释为Joint单独造成全部退化。姿态报告为 `eval-update180-sdpose-cpu-seeded-v1/evaluation.json`。

正式报告都在服务器 `report/pilot300-native-fullgpu-v2/`：`completed.json`、`eval-fixed-v1/evaluation.json`、`eval-sdpose-cpu-seeded-v1/evaluation.json`、`eval-identity-v1/evaluation.json` 和 `eval-detected-identity-v1/evaluation.json`；每项绑定 checkpoint、输入／输出 SHA 和执行实现。最终 checkpoint 为 `output/training/checkpoints/update_00000300_42e8d92b75644c299b27b872cfcccd3f.pt`，SHA256 `3744f6322a7102dda33b6a30bda88b5701aaf16085e4c0e6c8029f21f0f87896`。

|阶段|真实更新|耗时（秒）|peak allocated（GiB）|执行 ID 更新次数|
|---|---:|---:|---:|---:|
|S0 Body|30|1537.8|54.53|4|
|S1 Head|30|1434.8|49.24|2|
|S2 PoseLink Full|120|5552.7|49.30|13|
|S3 Joint|120|6122.7|57.50|7|

四阶段全部保留完整 ID 监督，均未使用 tiling／Hidden 暂存。训练与 GPU 固定噪声评价结束后显存已释放。先前 50% 显存预算的尝试均为 0 次更新，失败记录保留如下：

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

S1 四路预检通过，用时 **561.1 秒**；累计60步 checkpoint 的 SHA256 为 `09b19daa36e4f429db2f2aa4c5ecde78dfe9493dddd53f5583c87438f2c5d8a0`。S2 预检通过，用时 **260.3 秒**；累计180步 checkpoint SHA256 为 `a633dcfd8d991b3302d8c4efb62193acb21939f377bb4f385a0ae60dafa9acdc`。S3 预检通过，用时 **246.0 秒**。各阶段父链、optimizer／RNG、冻结状态和真实更新数均已校验，没有重置为新训练冒充连续300步。

`identity-positive-control-v1/evaluation.json` 验证真实 GT 的对齐路径：训练两人 Source–GT 为 **0.83778／0.83196**，观察两人为 **0.86517／0.80939**，相对准备阶段最大差 **0.00347**；人物槽位与 Canvas 一致。生成图的分数同时受 GT 位置对齐误差与身份还原影响，需要结合图像判断。

训练源码备份：`/data/shangguanrz/code-snapshots/native-v7-pilot-runtime-20261121T0330Z/runtime.tar.gz`，SHA256 `9fa97a5087d18b34d009856072391a7d8a57a1d8a69c4b05d91dcd33a5abe9e5`。本轮启动时服务器 Git HEAD 为 `00fb054d`，使用已审计 V7 源码覆盖层；任务结束时其他工作已将远端 HEAD 推进到 `463bcce`。实际训练版本以 checkpoint 内 code identity 与源码备份为准，不用单个 Git HEAD 代替执行证据。

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

查看结果：服务器 `output/pilot300-native-fullgpu-v2/baseline/` 有训练前4张对照，`final/update_000300/` 有训练／观察各一张20-step／seed42最终图；所有生成报告均记录 `target_rgb_used_as_condition=false`。本机 `output/pilot300-native-fullgpu-v2-local-preview/` 保存拉回的 Source、GT、初始／最终图和评价 JSON，原生像素未改写。四阶段 checkpoint 在服务器 `output/training/`。

用最终权重复跑推理（GPU需足够空闲，output／report 必须是新名字）：

```bash
$PY "$RUN/scripts/run_training.py" generate \
  --config "$RUN/configs/pilot300-native-fullgpu-v2/final-generation.json" \
  --checkpoint "$RUN/output/training/checkpoints/update_00000300_42e8d92b75644c299b27b872cfcccd3f.pt" \
  --output "$RUN/output/review-native-300-new" \
  --report "$RUN/report/review-native-300-new.json" --memory-fraction 0.9
```

复测评价时必须将下面已存在的 report-dir 换成新名字；GPU 评价与训练错开。命令只评最终两张图，与已有 seeded-v2 基线比较：

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
  --report-dir "$RUN/report/$NAME/eval-sdpose-cpu-seeded-v1"
$PY "$EVAL" --action identity-metrics --config "$CFG" \
  --generation-report "$FINAL" \
  --prepared "$RUN/data-v2/prepared.json" \
  --identity-model /home/shangguanrz/project/pic-edit/models/face_eval/identity_encoder.ts \
  --report-dir "$RUN/report/$NAME/eval-identity-v1"
```

GT 姿态控制可单独复测：在同一 SDPose CPU 环境运行 `audit_pose_control.py --config "$CFG" --evaluator "$EVAL" --report-dir <新的绝对目录>`。这只读取真实 GT，不把 Target RGB 用作生成条件。

真实检测五点身份审计（CPU；报告目录须换新名字）：

```bash
env CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  $PY "$RUN/scripts/audit_detected_identity.py" --config "$CFG" \
  --baseline-report "$RUN/report/$NAME/baseline-generation.json" \
  --final-report "$FINAL" --prepared "$RUN/data-v2/prepared.json" \
  --identity-model /home/shangguanrz/project/pic-edit/models/face_eval/identity_encoder.ts \
  --detector /home/shangguanrz/project/pic-edit/models/antelopev2/scrfd_10g_bnkps.onnx \
  --report-dir "$RUN/report/$NAME/eval-detected-identity-v1"
```

固定噪声比较使用相同 Joint 目标、sigma 0.2／0.5／0.8、每样本 noise seed 24042。姿态由 SDPose-Wholebody + YOLO11x 提取，score≥0.30，`order=geometry`，只比较 Body21／Body18 对应的 12 个身体关节；未检测到两人会显式记录，不能从均值里丢掉。身份指标是在目标五点位置对齐后的 Source→Generated 2×2 余弦矩阵，位置不准也会影响分数，需要结合姿态和图像解释。

私有脚本回归：`python -m pytest tests/test_native_v7_singlepair_preparation.py tests/test_native_v7_singlepair_driver.py tests/test_native_v7_singlepair_eval.py -q`，**55 passed、1 skipped**。跳过项为本地 CPU-only 机器上的真实 CUDA 暂存测试，之前已在服务器核验输出/梯度逐值一致。服务器可用 `V7_PILOT_SCRIPT_ROOT` 指定实验脚本路径后复测。

## 与其他实验的关系

上游为 Native V7 架构（`1838e58`）和 Hi4D r4 grounded binding pilot 的两条 typed 4Ref 数据。本实验补齐训练资格与 repair，验证真实7B训练、连续阶段与生成闭环。

实际训练的 `native_v5/losses.py:93,101` 将 Target GT 的特征传给 ID 与对比项，V7 callback 沿用该 reference；Source 特征用于条件调制，当前并无直接 Source→Generated 的绝对身份一致性项。训练相对 GT 的 ID Loss 变好，不保证 Source 身份相似度改善。下一步优先对照 Source 绝对一致性、有效 ID 更新频率和各组 Joint 学习率／冻结策略，同时检查背景约束；这是根据现象提出的排查方向，不是已验证的单一原因。

优先补齐Source身份与背景约束后，用同一双人 Source 的3–4个目标姿态测试条件控制，再扩到8–32个双人配对并按人物划分验证。背景与手部细节、跨人物泛化和各模块消融仍需单独评价。本实验没有启动大规模训练，全部GPU与CPU评价任务均已退出。
