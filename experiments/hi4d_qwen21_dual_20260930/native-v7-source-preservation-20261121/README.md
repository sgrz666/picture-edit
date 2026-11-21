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

## 当前结果

真实CPU准备与三份data检查已通过：两条样本各2个Source脸，背景token面积分别3683.75／3715.48047；Source特征只从Source重新提取。`source-background-grad-v1.json`的真实7B完整反向通过，291.9秒，peak allocated59,954,979,840字节。补充CLI纯velocity消融关闭Source目标后，须以最终代码重新取得v2报告再训练，不复用不同code identity的旧gate。

旧配置record/signature逐值兼容、单脸／空脸／GT+Source共用一次decode的规格探针和质量审查通过。V7相关本地回归213 passed／1 skipped；旧V5 Windows有原POSIX绝对路径限制的9个失败，未修改该旧模块，服务器Linux同套件24 passed／1 skipped。GPU新训练／生成效果待本轮报告，不预先宣称身份或背景改善。

## 关系

上游为`native-v7-singlepair-overfit-20261117`的真实输入、缓存与阶段权重；新目标默认关闭，旧V7配置record／signature不变。本实验不启动大规模训练，不覆盖旧结果。
