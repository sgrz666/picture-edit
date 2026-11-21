# Native V7 Source 身份与背景保持

## 目的

上一轮300步姿态改善，但检测五点对齐的Source身份与背景保真下降。新实验复用原生4Ref、同一训练／观察配对和180步已提交权重，显式加入Source绝对cosine和Source背景x0约束；不改变Qwen或PoseLink架构。

## 内容与使用

`scripts/prepare_assets.py`只从真实Source检测五点并用同一固定SHA的TorchScript提取Source特征，按source_indices映射到target人物槽位；背景排除Source／Target人物并集膨胀16px和padding。输出到新的`data/assets-v1/`，生成control、source、source-background三份配置到`configs/pilot-v1/`，每份默认12步。

准备脚本首轮复用原双人唯一匹配门禁，要求两张Source脸均合格，缺失时整体拒绝准备；底层目标可处理单有效Source／GT交集，不能用该能力放宽此实验的检测资格。

服务器根目录：`/data/shangguanrz/pic-edit-experiments/qwen21_hi4d_dual/native-v7-source-preservation-20261121-v1`。旧数据、缓存、180步权重只读复用。所有配置保持相同seed／学习率／预算，并显式统一ID gate为每步、sigma≤0.5；因此control是新gate下的对照，不等于旧300步策略。

```bash
PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
ROOT=/data/shangguanrz/pic-edit-experiments/qwen21_hi4d_dual/native-v7-source-preservation-20261121-v1
OLD=/data/shangguanrz/pic-edit-experiments/qwen21_hi4d_dual/native-v7-singlepair-overfit-20261117-v1
$PY "$ROOT/scripts/prepare_assets.py" --root "$ROOT" --baseline-root "$OLD" --budget 12
```

准备脚本拒绝覆盖既有资产与配置，重做需使用新的实验root。训练前必须真实data/grad预检；不把新增loss或预检当作生成效果改善。

`scripts/run_pilot.py`运行control与source-background两组各12步，总共24个新增optimizer updates；每组均独立从180步父权重继续，核验梯度门禁、父链、准确更新数和commit，然后各生成两张seed42／20-step图。source-only配置保留，未自动增加第三组训练。

```bash
env LD_LIBRARY_PATH=/data/shangguanrz/仪表盘/compat OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  $PY "$ROOT/scripts/run_pilot.py" --root "$ROOT" --baseline-root "$OLD" \
  --name paired12-v1 --source-gradient "$ROOT/report/source-background-grad-v2.json" \
  --memory-fraction 0.8 --detach
```

启动前`source-background-grad-v2.json`必须是当前代码／配置的真实通过报告；已有run名不得重复。实际PID与完整命令在`logs/paired12-v1-pid.json`，每组日志独立保存。

查看进度与当前Source loss分项：`$PY "$ROOT/scripts/status.py" --root "$ROOT" --name paired12-v1`。最终每臂的checkpoint与生成config／report SHA在`report/paired12-v1/completed.json`；两组均为累计192步，不能混称旧300步结果。

## 当前结果

真实CPU准备与三份data检查已通过：两条样本各2个Source脸，背景token面积分别3683.75／3715.48047；Source特征只从Source重新提取。`source-background-grad-v1.json`的真实7B完整反向通过，291.9秒，peak allocated59,954,979,840字节。补充CLI纯velocity消融关闭Source目标后，须以最终代码重新取得v2报告再训练，不复用不同code identity的旧gate。

旧配置record/signature逐值兼容、单脸／空脸／GT+Source共用一次decode的规格探针和质量审查通过。V7相关本地回归213 passed／1 skipped；旧V5 Windows有原POSIX绝对路径限制的9个失败，未修改该旧模块，服务器Linux同套件24 passed／1 skipped。以下结果来自本轮已完成的两组真实训练与生成。

最终CLI版本的`source-background-grad-v2.json`预检通过（312.3秒，peak allocated59,954,979,840字节）；CLI `velocity_only`已明确关闭Source目标，旧v1报告保留但未复用于新代码训练。新目标与V7／V62 RL相关回归122 passed／1 skipped，包含59个kernel、31个资产、7个训练集成和7个对照公平性测试。

两臂各12步及生成已完成，SourceBG训练876.8秒、peak allocated61,740,431,360字节；4个有效Source身份步骤、全部12步背景项实际参与训练。两个checkpoint：Control `fab153f7572942e715fe836c588ceff60094a3aa831fb204ccc35e03777906ad`，SourceBG `eca8f5ece61683b8d00871211d196c328de4af99eaf4696902df76eeaf1e3f02`，父均为180步`a633dcfd…9acdc`。首步同一sigma=.246808、原始loss一致，新增Source ID=.2345085、背景=.0778537，梯度非零。

CPU最终评价均exit0。所有图像都检出2人、24/24共同点有效，人物匹配无歧义；SDPose每次检测seed24042、pipeline SHA、阈值／坐标顺序与旧试验一致。以下均为**本轮Control192→SourceBG192**，不与旧300步混比：

|样本|检测五点对齐Source ID（两人）|PCK@0.2|关节误差px|背景MAE|背景PSNR dB|
|---|---|---:|---:|---:|---:|
|训练|[.56429,.69608]→[.62219,.72833]|95.83%→95.83%|15.1689→15.2263|5.14754→5.11929|27.15641→27.17675|
|观察|[.62175,.86348]→[.63391,.87256]|95.83%→100%|17.2431→17.0453|4.85250→4.82460|27.67629→27.87691|

四个人脸身份分数均提高，均未触发swap；背景两样本有小幅改善。训练姿态误差略升0.057px，不能宣称完全无损。逐人物还存在训练slot0误差+0.187px、观察slot1+0.018px，其余两人下降，完整值保留于JSON。这里只是同一Source、1个训练配对、12步的正向证据，不证明跨人物泛化、长程稳定或身份／背景已完全恢复。

报告：`report/paired12-v1/eval-paired-detected-identity-v1/evaluation.json`（真实SCRFD五点与Source身份）和`eval-paired-sdpose-cpu-v1/evaluation.json`（姿态／背景与4张叠图）。`baseline`／`final`字段分别表示Control192／SourceBG192，两者均有确切权重SHA。原始图在`output/paired12-v1/{control,source-background}/update_000192/{train,observation}/`；本机拉回副本及JSON在`output/local-preview/`。全部实验GPU／CPU进程均已退出，没有增加第三组或更多训练。

复用本轮SourceBG权重推理（GPU需足够空闲，输出和报告换新名字，不改原结果）：

```bash
$PY "$OLD/scripts/run_training.py" generate \
  --config "$ROOT/report/paired12-v1/source-background-generation.json" \
  --checkpoint "$ROOT/output/training/checkpoints/update_00000192_19203500495047c59007c90e347049af.pt" \
  --output "$ROOT/output/review-sourcebg-new" \
  --report "$ROOT/report/review-sourcebg-new.json" --memory-fraction 0.8
```

在其他训练配置启用新目标须同时绑定`preservation={"source_identity":0.2,"background":0.1}`和相应`preservation_index={path,sha256}`；索引必须由该数据的Source真实五点和同一evaluator生成，不能直接套用本实验index到另一张Source。默认None保持旧行为，网络结构与原V7权重格式不变。

复测CPU评价可复用旧实验`$OLD/scripts/evaluate_run.py --action image-metrics --pose-device cpu`，config使用`$ROOT/report/paired12-v1/source-background-generation.json`，generation-report传两组的`control-images.json`与`source-background-images.json`，必须换新的report-dir。环境使用既有SDPose venv，`CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_HOME=/home/shangguanrz/nihaochen/envs/qwen21`。身份审计用`$OLD/scripts/audit_detected_identity.py --help`列出的参数，baseline/control与final/SourceBG对应上述两个generation报告，prepared仍是`$OLD/data-v2/prepared.json`，TS与SCRFD路径同旧试验。

代码回归：`python -m pytest tests/test_native_v7_preservation.py tests/test_native_v7_preservation_assets.py tests/test_native_v7_preservation_training.py tests/test_native_v7_preservation_pilot.py tests/test_native_v7_training.py tests/test_native_v62_rl.py -q`，实际**122 passed、1 skipped**；跳过项为需额外资源的CUDA测试。

## 关系

上游为`native-v7-singlepair-overfit-20261117`的真实输入、缓存与阶段权重；新目标默认关闭，旧V7配置record／signature不变。本实验不启动大规模训练，不覆盖旧结果。
