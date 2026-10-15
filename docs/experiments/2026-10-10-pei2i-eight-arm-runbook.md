# TikTok PE-I2I Body＋Face 八组实验：修复后的运行手册

本手册对应 `docs/superpowers/plans/2026-10-10-pei2i-body-face-eight-arm-code-plan.md`，运行数据与模型以本次 `data/lock.json` 为准。不要沿用旧手册中错误的初始化 SHA、学习率或“PB 是外观提示词”的描述。

## 1. 工作目录与当前产物

服务器：SSH `sg`，实际主机名 `node01`。独立实验目录：

```text
/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009
```

运行代码在该目录的 `code/`；Python 为 `/home/shangguanrz/nihaochen/envs/qwen21/bin/python`。`code/datasets`、模型视图及旧实验路径映射指向现有只读资产。原项目的训练结果不需要移动或删除。

修复启动时，上述 PE 实验目录在当前 `sg` 上不存在；本次创建了独立目录。旧的 `bfs_body_face_joint_pilot_20261008` 确实存在，用其真实数据与 J256 权重继续实验。`repair_audit/environment.json` 记录了这一事实，不能把本次新建结果描述成已经验证了原 PE 实验结果。

状态查看：`repair_audit/*_launch.json` 保存进程启动参数、状态、结束时间和退出码；详细过程见 `logs/`。`running` 只是启动状态，`complete`＋退出码 0 也需结合该阶段实际产物核验。

本地图片与中文报告保存在：

```text
D:/codeplus/pictureedit/research/pei2i_body_face_8arm_20261009
```

## 2. 比较什么

八组是 `qwen_{p0,pg,pb,pbg}` 与 `head_{p0,pg,pb,pbg}`。

| 协议 | PE 实际观察的图片 | Qwen 收到的 RGB |
|---|---|---|
| P0 | 不调用 PE，使用历史固定短提示词 | A |
| PG | A、B depth、B normal、过滤后的 B DWPose | A |
| PB | A、B | A |
| PBG | A、B、B depth、B normal、过滤后的 B DWPose | A |

两套底座分别为固定 Qwen-Image-2.1 和同一 Qwen＋冻结 BFS Head V1.1，Head 强度 1.0，从本研究的第 0 次更新就挂载。每组从同一个 J256 Adapter 出发，重新创建 AdamW，然后执行 **1024 次额外更新**；不在 Qwen 组再额外预训练一次。

这是“已有联合 Adapter 在不同文本/底座下继续适配”的比较。共同 J256 曾在纯 Qwen 上训练，会影响初始偏好；本实验不回答随机初始化 Adapter 下各底座的最优性能。BFS Head 原始任务通常使用两张参考 RGB，本研究严格统一为 A 单图输入，结论仅适用于这个输入协议。

## 3. 图像、几何与 GT 边界

- RGB 保留原始像素宽高。VAE 输入只补到 16 的倍数，例如 604×1080→608×1088；结果去补边回 604×1080。视觉编码器内部预处理另行记录。
- Body 分支使用 normal、Body25、part、26 维 SMPL-X global、人体 mask；Face 分支独立接收目标 72 点、jaw、expression 和 A 的身份特征。26 维并不是全部面部输入。
- 本轮 Depth 与 DWPose 用于 PE 输入和可视化，**没有启用 Depth Adapter**。Hand、Interaction、共享 Detail 关闭；Body＋Face 联合训练。
- A 提供身份、发型、服装和场景；B 提供身体姿态、头部朝向、可观察表情及手部配置。服装纹理保持，但褶皱和姿态阴影允许变化。
- B RGB 可以给 PB/PBG 的 PE。Qwen 的图像参考条件仅为 A；提示词需要编译成单独可理解的姿态描述。
- 训练时 B 仍是监督目标：会编码 B 得到 flow 训练目标/加噪训练状态，并计算训练损失。这与把 B 当参考图喂给 Qwen 是两件事。禁止把测试过程描述成“完全没有读取 GT”；评测可以离线读取 GT 算指标，但不会反向传播。
- 验证、跨人物诊断、最终测试只推理，禁止 optimizer step、backward 和训练损失调用。跨人物诊断禁止整图 B GT 像素误差。

## 4. 不变的预算与数据

真实资产 SHA256：

```text
J256:   9e0f048c585937ab74be792eb6d9730b0621eccda16e1744a6837b0eb9d24651
train:  854133776d2067aa392442553f3003ef9e6d45107ccc9cffe5a4af8600fb929f
val:    cece883b96dbbc9270fd77d4bb1e973470f52f941af85c2a0746947f5d91e10a
Qwen revision: 790c92633540aa0cb11d9abf19eb46d861714758
```

数据：训练 512 对、验证 24 对、固定跨人物诊断 6 对、独立测试 24 对。按原视频组隔离，测试候选只按输入资产完整性筛选，不能根据输出好坏换样本。

每组 batch=1、无梯度累积；FP32 Adapter 与优化器状态，BF16 autocast；warmup 20 步。Body 三类参数学习率为 `2.5e-5/1.25e-5/2.5e-6`，Face 为 `5e-5/2.5e-5/5e-6`，具体组名以 `config.json` 和 optimizer 分组为准。flow＋0.5 face ROI＋0.1 identity＋0.05 perceptual；辅助项每 4 步检查一次，并受真实 Face 有效性及 sigma 门控。

检查点为 0/128/256/384/512/640/768/896/1024。推理 30 步、CFG=1、Body=1、Face=1、Hand=0。

| 内容 | 计划 |
|---|---|
| train monitor | 固定 8 对，9 个检查点，seed 42 |
| 完整 val | 24 对，在 0/256/512/768/1024，seed 42、43 |
| 中间 val monitor | 固定 9 对，在 128/384/640/896，seed 42 |
| 跨人物诊断 | 6 对，在 0/512/1024，seed 42 |
| 最终 test | 24 对，在 0/1024，seed 42、43，八组训练完成后统一开放 |

总计 462 次推理/组、3696 次推理；最终测试前 2928 次，最终测试 768 次。正式 PE 重写 566×3=1698 条，P0 不计入 PE 数量。QA8 的 24 条合格记录按完全相同的请求签名复用，不重复生成。

## 5. 按顺序运行

已完成的锁定/Face 阶段不要随意重建；模型、模板、Face 缓存或数据名单变化会使后续签名失效。先检查 launch 记录和产物。以下命令用于新实验或按阶段续跑，不能与已在运行的同阶段任务重复启动。

```bash
ssh sg
cd /data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009/code
PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
RUN=/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009

# 只预览，不创建文件、加载模型或启动训练
$PY scripts/run/run_qwen21_pei2i_study.py --run "$RUN" --phase train --resume --dry-run

# 新目录准备阶段；本次 lock 与 faces 已记录完成
$PY scripts/run/run_qwen21_pei2i_study.py --run "$RUN" --phase lock
$PY scripts/run/run_qwen21_pei2i_study.py --run "$RUN" --phase faces

# 先生成前8对×3协议，检查24条真实输出
$PY scripts/run/run_qwen21_pei2i_study.py --run "$RUN" --phase rewrite-qa
```

QA 检查需要逐条对照输入图片、raw reply 和 compiled prompt：姿态是否读对、是否带入 B 的服装/背景、是否把 normal 颜色当皮肤、是否臆造不可见手指/腿、编译后是否仍依赖“图2”等不存在的输入。统一模板若修改，应保留旧 QA，再重做全部 QA；禁止只手改个别样本让其看起来更好。

实际检查后，用同一代码版本保存签名审核（`notes` 应记录实际发现）：

```python
from pathlib import Path
from scripts.precompute_qwen21_pei2i_prompts import qa8_review_provenance
from src.pose_control.qwen21_full.prompt_study.prompts import request_sha256
from src.pose_control.qwen21_full.prompt_study.protocol import atomic_json
run = Path('/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009')
payload = qa8_review_provenance(run)
payload.update(reviewed=True, notes=['逐条检查后填写真实结论'])
payload['review_signature_sha256'] = request_sha256(payload)
atomic_json(run / 'prompts/qa8.json', payload)
```

完成 QA 后继续：

```bash
$PY scripts/run/run_qwen21_pei2i_study.py --run "$RUN" --phase rewrite --splits train val diagnostic
$PY scripts/run/run_qwen21_pei2i_study.py --run "$RUN" --phase cache --splits train val diagnostic
$PY scripts/run/run_qwen21_pei2i_study.py --run "$RUN" --phase selfcheck
$PY scripts/run/run_qwen21_pei2i_study.py --run "$RUN" --phase preflight
$PY scripts/run/run_qwen21_pei2i_study.py --run "$RUN" --phase initial
$PY scripts/run/run_qwen21_pei2i_study.py --run "$RUN" --phase train --resume
$PY scripts/run/run_qwen21_pei2i_study.py --run "$RUN" --phase finaltest
$PY scripts/run/run_qwen21_pei2i_study.py --run "$RUN" --phase report
$PY scripts/run/run_qwen21_pei2i_study.py --run "$RUN" --phase export
```

Preflight 检查所有八组的低/高 sigma Body/Face 梯度、冻结状态，以及独立目录中 10 次私有更新的恢复测试。恢复检查结果在 `preflight/resume_check/result.json`；数值漂移超出实际重复运行范围时需要调查，不要改门禁强行放行。私有更新不计入 8192 次正式更新。

## 6. 恢复与故障处理

- `train --resume` 从经过验证的正式检查点继续；保留已完成的图片，不重复计数。
- 别用 `--steps 256` 缩短本次正式训练，再当作 1024 步比较；本次预算固定。
- 有失败推理时，目录内的失败记录会保留。明确重跑应使用新的输出目录，不能直接删除失败行或替换样本。
- 如果进程中断在写 metrics 后、保存 checkpoint 前，先查看恢复检查提示。不能只删最后若干 loss 行或用较早 checkpoint 抹掉已提交训练。
- `nvidia-smi` 的 NVML 版本错误不能代表 CUDA 计算不可用；本机已经通过真实 PyTorch CUDA 计算。不要为此重启服务器。
- PE 必须真实加载本地权重，采用官方 system prompt、thinking、24000 token 上限和记录的采样参数。没有模型或解析失败时应报错并保留原回复，不允许模板模拟成功。

## 7. 测试代码与查看结果

在运行代码目录执行：

```bash
$PY -m pytest tests/test_qwen21_prompt_study*.py -q
```

CPU 测试覆盖输入隔离、原尺寸、缓存来源、恢复轨迹、评分与报告契约；GPU preflight 和首个真实推理仍是正式运行的必要验证，CPU 通过不能代替生成效果评估。

中文网页为 `report/index.html`。每条结果保留整图、A/B 输入、normal/depth/DWPose、固定目标 Face 和左右 Hand ROI、提示词及其输入、原始 PE 回复、模型/检查点 SHA、seed 和失败信息。网页生成时可以仍是“进行中”，不得把空白占位当作成功。

背景指标使用 A/B 原尺寸人物 mask 并集扩张 8 像素，再排除画布四边 8 像素；画布不匹配或背景无效则缺失＋原因，禁止 resize 后计算。两次推理 seed 先按 pair 求有效均值，再按原视频组求宏平均；公开缺失和检测失败数量。48 次推理不是 48 个独立人物，也不是两个独立训练重复。

历史 J/Qwen/BFSHead 图片另列，标明旧 prompt、预算和 LoRA 加载方式，不计入正式实验数量，不与新实验直接宣称公平优劣。

导出只包含文本、图片和网页，排除模型、`.pt` 缓存和权重。下载 `export/results.tar.gz` 后，在独立本地目录验证 `files.json` 的 SHA 及所有网页相对链接，再更新本地展示目录，避免覆盖旧可视化证据。
