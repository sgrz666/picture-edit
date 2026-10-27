# TikTok Body + Face 联合训练与冻结 BFS 底座对照

> 状态：旧 512 协议已停止。用户最终要求保留原始像素宽高，相关改造尚未实现；后续请先读同目录 `2026-10-08-bfs-body-face-joint-handoff.md`。下文描述历史配置和历史结果的查看方法。

本实验沿用已有 Body6000 / Face3000 Adapter 初始化，在 512 个 TikTok 训练对上做短程联合训练。原始实验与数据没有覆盖。实际结果、已完成步数及失败记录以本地导出的 `report/summary.md` 和各阶段日志为准。

## 如何看结果

在本地打开：

- `D:/codeplus/pictureedit/research/bfs_body_face_joint_pilot_20261008/report/index.html`：按样本查看 A、B、几何图以及各模型的输出和面部裁剪。
- 同目录 `summary.md`：完整 36 个验证对的指标、配对差值及视频 / 人物组汇总。
- `evaluation/`：每个阶段独立保存原始 PNG、脸部 PNG、对比 JPG、逐样本记录、评分和配置。B 图只用于离线对照及监督。
- `quality_reviews/`：中途 12 个固定样本的全身和面部对比大图，以及继续训练前的人工审查。
- `data/lock.json`：训练样本、验证样本、原始视频组、固定监看下标和哈希。
- `data/runtime_source_20261008.tar.gz` 和 `runtime_source_hashes.json`：sg 上实际执行的代码快照与逐文件哈希。远端评估脚本 SHA256 为 `45f09c6258a19375ba5fb444b8757e0db0406c8b91abb2633d8bd2fe90f2159b`；本地源码另有三处显式 UTF-8 读取，结果协议相同。

HTML 是本地静态报告，保留整个实验文件夹后可直接用浏览器打开。所有输出仍按阶段单独保存；对比图缩放不改变原始生成 PNG。

## 对照设计

| 阶段 | 底座 | 训练内容 | 新增步数 |
| --- | --- | --- | ---: |
| J | 原 Qwen-Image-2.1 | Body + Face 联合训练，前 256 个训练对 | 256 |
| Qwen | 原 Qwen-Image-2.1 | 从 J256 开始，后 256 个训练对 | 256 |
| BFSBody | Qwen + 冻结 BFS Body V1.0，强度 1 | 从同一 J256 开始，后 256 个训练对 | 256 |
| BFSHead | Qwen + 冻结 BFS Head V1.1，强度 1 | 从同一 J256 开始，后 256 个训练对 | 256 |

正式预算共 1,024 次优化器更新。三个最终模型都经历 512 对新增训练，后半段分别使用全新的相同优化器，不继承其他分支的训练状态。梯度探针不做优化器更新；独立的恢复诊断共 18 次私有更新，另存于 `preflight/`，不计入正式训练。

训练只开放 Body 和 Face 参数。Qwen、BFS LoRA、Hand、Interaction 和 sharedDetail 保持冻结。Body 全局 26 维继续沿用原定义；B 的 jaw / expression / face 特征通过独立 Face 分支传入。A 提供 RGB 和身份特征，B 提供几何条件。此对照没有使用 BFS 官方的两张 RGB 输入协议，因此衡量的是它在当前 Adapter 协议下的适用性。

Body 学习率为原设置的 0.25 倍，Face 为 0.5 倍，线性预热 20 步。损失为 `Body flow + 0.5 Face ROI flow + 0.10 identity + 0.05 perceptual`；身份 / 感知项只在每 4 步且低噪声、有效面部的条件下使用。Face 沿用时间门控，高噪声时关闭；Body 全程工作。没有额外使用冻结 Body 教师约束。

## 验证与解释边界

固定 24 个 TikTok 验证对和 12 个 iPER 诊断对；TikTok 训练 / 验证按原始视频组隔离。每组保存训练前 12 图、128 步时 12 图及最终 36 图；共同 J 保存初始 / 最终全 36 图及中途 12 图。另有 4 个 Body-only 和 4 个 Face-only 推理诊断。

身份指标为生成图与 A 的唯一人脸 ArcFace 相似度；姿态指标将生成姿态与原始 B 的几何关键点比较。缺失人脸或生成失败保留在记录中，不用零值填补身份分数。比较使用相同有效样本的配对差值，视频 / 人物组也单独汇总。TikTok 验证对 9 的 Face 条件无效，仍保留并标注。

这些是短程、单随机种子的开发验证，包含同人跨帧样本，不能证明跨人物身份迁移，也不能证明联合训练优于等预算 Face-only 训练。iPER 是额外数据集诊断，不应声称所有人物从未被旧初始化见过。部分 CUDA 算子仍有浮点波动；微小分数差不作为确定优劣的依据。表情、头部方向、服装和背景需要结合保存图像判断。

## 复用与补看

远端仓库：`/home/shangguanrz/project/pic-edit`。

远端实验：`/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008`。

权重留在远端 `training/{J,Qwen,BFSBody,BFSHead}/checkpoint_step_000256.pt`；本地导出保留图片和小文件，不重复下载大型检查点。

重新计算保存图像的指标（不重新生成）：

```bash
cd /home/shangguanrz/project/pic-edit
PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
RUN=/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008
$PY scripts/eval/eval_qwen21_joint_pilot.py --score-only --output "$RUN/evaluation/Qwen_final_tiktok"
$PY scripts/reports/summarize_qwen21_joint_pilot.py --run "$RUN"
$PY scripts/reports/build_qwen21_joint_pilot_report.py --run "$RUN"
```

Windows 下载最新图片和报告：

```powershell
& D:/codeplus/pictureedit/scripts/run/sync_qwen21_joint_pilot.ps1
```

若需要给任一检查点增加推理样本，使用 `scripts/eval/eval_qwen21_joint_pilot.py --help`，指定检查点、数据 / 几何缓存、底座及正确 LoRA，把 `--output` 设为新的独立目录。用保存的 `config.json` 对照参数。不要复用已有正式阶段名称改变锁定样本；不同数据或采样配置的补充测试应保存到独立实验根目录后单独构建报告。

训练入口 `scripts/run/run_qwen21_joint_pilot.py --phase ...` 使用固定样本和梯度审查。正式阶段已经存在时不要重复启动；128 → 256 的续训要求先审查全部 12 个固定监看样本，再填写对应的质量审查记录。
