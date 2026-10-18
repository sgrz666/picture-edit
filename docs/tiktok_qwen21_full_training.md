# TikTok → Qwen 2.1 完整版训练使用说明

2026-10-03，验证主机为 `ssh sg`。项目：`/home/shangguanrz/project/pic-edit`。
Python：`/home/shangguanrz/nihaochen/envs/qwen21/bin/python`。

## 目前适合什么训练

适合 512 分辨率的小规模 Body 试训，以及受控的 Face 链路验证。
暂不建议直接进行大规模 Body＋Face 长训：本次只验证了 4 个训练对、2 个验证对；
完整生成的目标姿态仍不准确，Face 尚未显示质量收益。先用固定验证集评估一段短训练，再决定扩大规模。

严格数据池原有 train/val/test 为 8,574 / 1,120 / 1,156 对，原视频组无交叉。
Body 条件进一步筛选后 train/val 为 8,548 / 1,112 对。排除 26 / 8 对身体有效关键点不足的样本。
原始图片、条件、数据清单不被修改。测试集只用于检查分组交叉，不进入本次训练、Face 特征提取或采样。

## 复现完整小规模链路

登录服务器后执行，使用一个新的输出目录：

```bash
cd /home/shangguanrz/project/pic-edit
bash scripts/run_tiktok_qwen21_full.sh smoke \
  /home/shangguanrz/nihaochen/tiktok_full_my_smoke
```

依次完成数据筛选、4/2 对的原生缓存、Body 更新与恢复、Source-only ArcFace/DINO 缓存、
Target 人脸辅助监督、Face 更新与恢复，以及 initial/body/body_face 三种模式的 30 步采样。
每个阶段失败都会停止。模型均使用服务器已有文件。日志在输出目录的 `logs/`。

查看 `data/readiness.json`、`body_smoke/summary.json`、
`face_stable_smoke/summary.json` 和两个分支的 `resume_verification.json`。
检查所有计数步都有非零梯度、冻结参数保持不变，以及参数/优化器恢复一致。
同时查看 `generation/images/`，不能用 loss 或“能保存 checkpoint”代替生成质量评价。

## 开始独立的 Body 试训

```bash
cd /home/shangguanrz/project/pic-edit
bash scripts/run_tiktok_qwen21_full.sh prepare \
  /home/shangguanrz/nihaochen/tiktok_full_pilot
bash scripts/run_tiktok_qwen21_full.sh body \
  /home/shangguanrz/nihaochen/tiktok_full_pilot 200
```

`body` 使用全部合格 Body 清单；首次执行会预计算所有 Source/Target 的原生缓存，耗时大于小样本验证。
这是一段试训，不代表完整数据已收敛。默认从已存在的 Qwen 完整版 Body checkpoint 初始化；
可以用环境变量 `TIKTOK_BODY_INIT` 指定同架构的其他 checkpoint。

继续到总计 400 步：

```bash
bash scripts/run_tiktok_qwen21_full.sh resume-body \
  /home/shangguanrz/nihaochen/tiktok_full_pilot 400
```

步数是最终累计 optimizer 步数。一般试训使用至少 20 步，以容纳默认 20 步的桥接输出 warmup。

## Body 验证后再准备 Face

```bash
bash scripts/run_tiktok_qwen21_full.sh face-prepare \
  /home/shangguanrz/nihaochen/tiktok_full_pilot
bash scripts/run_tiktok_qwen21_full.sh face \
  /home/shangguanrz/nihaochen/tiktok_full_pilot 200
bash scripts/run_tiktok_qwen21_full.sh eval \
  /home/shangguanrz/nihaochen/tiktok_full_pilot
```

`face-prepare` 才会为完整合格清单提取 Face 缓存，并产生额外的人脸可训练清单；
本次交付的实际 Face 缓存只覆盖 4/2 对验证样本，不能冒充全量缓存。
该阶段可按相同参数重新执行并复用已完成的序列缓存，输入图片和几何哈希改变时会拒绝复用。
不要将小样本输出目录直接用于全量 Face 准备；应使用上述独立 pilot 目录。

Face 默认采用 ROI 归一化 flow loss、`[-1,1]` evaluator 输入、五点辅助对齐，
三组学习率为 `1e-6 / 5e-7 / 1e-7`。这是保守的试训起点，不是已验证的最优参数。
此前 `1e-4 / 5e-5 / 1e-5` 在两个实际样本更新后使固定噪声 Face loss 从 0.437 升到 2.792。
继续 Face：

```bash
bash scripts/run_tiktok_qwen21_full.sh resume-face \
  /home/shangguanrz/nihaochen/tiktok_full_pilot 400
```

数据清单、对齐方式、seed、prompt、学习率和损失配置保持一致。改变这些设置时使用新的输出目录和
`--init-checkpoint`；原生训练入口的 `--resume` 会拒绝已检查的配置不一致。
Face 缓存须作为训练输入固定保存：内容指纹检查可以检测损坏，但目前 checkpoint 尚未绑定完整 Face 缓存内容哈希。

## 数据与分支约束

- 完整版 TikTok 使用 `resize`：RGB、normal、part、人体 mask、body keypoints、Face 几何共享坐标。
  完整版暂不支持 TikTok `square_crop`。保留整幅竖屏后方形缩放存在外观比例变化，需要后续质量实验评估。
- 缺少关键几何条件、空 part/person mask、非有限参数、有效身体关键点不足会报错，不再静默补零。
- 身份和纹理由 Source 提供；Target-only 帧的 ArcFace/DINO 数组为零。
  Target RGB 只进入训练目标和离线辅助损失，采样只编码 Source 图像。
- 本次打通 Body→Face。Hand、Detail、Interaction 未参与这条单人训练链路。
  原视频组隔离不等于已经完成人脸身份去重认证。
- `eval` 默认只生成一个验证对，使用相同 seed、30 步、`progress-mode=sigma` 和 checkpoint Face grid。
  扩大质量评估时直接使用 `eval_iper_qwen21_full.py --max-samples ...`。

## 验证命令

Windows 工作区执行：

```powershell
& D:\codeplus\pictureedit\research\tiktok_training_20261003\verify_remote.ps1
```

此次远端相关回归 **188 项通过**；本地 20 项新增测试通过。已有 SWIG 的弃用警告不影响结果。
不依赖本机的 Qwen 环境或模型文件。
