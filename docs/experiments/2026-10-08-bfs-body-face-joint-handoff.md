# Body + Face / BFS 实验交接（2026-10-08）

用户接手后续实现和实验；代理已停止训练、推理和子代理。此交接合入已有实验实现和下一步方案，**原始像素宽高训练尚未实现**。

## 工作目录

- Windows 主工作目录：`D:/codeplus/pictureedit`，分支 `main`。
- 实际训练服务器：`ssh sg` 后进入 `/home/shangguanrz/project/pic-edit`，分支 `main`。
- Python：`/home/shangguanrz/nihaochen/envs/qwen21/bin/python`。
- 旧实验权重与结果：`/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008`。
- 本地图片与报告：`D:/codeplus/pictureedit/research/bfs_body_face_joint_pilot_20261008`。

本次变动包含冻结 BFS 加载、Body + Face 联合损失 / 梯度和冻结审查、固定数据划分、分阶段训练、逐样本推理导出、评分、HTML 报告与恢复检查。配套现有 Qwen21 runtime 一起交接。旧实验协议见同目录 `2026-10-08-bfs-body-face-joint-pilot-runbook.md`。

## 已完成的旧协议结果

旧训练使用 TikTok letterbox 到 512×512：保留宽高比，但仍然等比缩小并补边。旧 iPER 诊断使用 resize 到 512×512。**这两者都不是用户最终要求的原始像素尺寸训练。** 保留这些结果作历史诊断，不并入新协议的主对照。

- J：从已有 Body6000 / Face3000 Adapter 初始化，联合训练 256 步，完整 36 对验证和中期 12 对验证已保存。
- Qwen：从同一 J256 初始化，独立新优化器训练 256 步，训练前 / 中期各 12 对及最终 36 对已保存。
- BFS Body V1.0：真实加载和梯度预检通过；保存了训练前 9 个 TikTok 输出。iPER 热加载只保存了 1 个成功输出，下一例因主动停止留下失败记录；这属于流程中断。未开始正式分支训练。
- BFS Head V1.1：未开始本次正式分支训练。
- 正式更新共 512 次；独立恢复诊断另有 18 次私有更新，不计正式预算。
- Body-only / Face-only 各 4 张诊断图已保存。Face-only 身份分数高，但明显保留 A 的姿态和构图，不能当成成功重定向。

旧实验根目录 `data/preprocessing_review_required.json` 记录停止原因，旧 runner 遇到此文件会直接退出，不再派发 GPU 任务。不要仅删除标记后继续旧 512 流程。

## 下一步方案：原始像素宽高

继续保留实验主线：原 Qwen、固定 Qwen + BFS Body V1.0、固定 Qwen + BFS Head V1.1，三组联合训练 Body + Face；Qwen / LoRA 和 Hand / Interaction / sharedDetail 冻结。

输入 A 与目标 B 使用原始像素宽高，允许不同分辨率、不同长宽比和 A/B 不同尺寸。若 VAE / Transformer 要求对齐，只向边缘补到真实要求的倍数；不缩放原始 RGB。输出裁掉补边，恢复 B 原始宽高。内部特征图或专用人脸编码器的固定输入与整图训练分辨率应分开记录。

建议复用已锁定的 512 个训练对、24 个 TikTok 验证对、12 个 iPER 诊断对，以及固定 12 对监看子集。数据可直接读旧 `data/lock.json` 和各 JSONL，但必须使用新缓存、新实验目录、新签名和新初始推理。不得把旧 512 的 J256 与新原始尺寸的 J256 混用作公平初始化。

每个最终候选仍采用共同 J256 + 独立 256 步，正式总预算 1,024 步；从旧 Face3000 初始权重开始一套新的共同训练。先用不同宽高的少量真实样本完成数据 / VAE / Adapter / 损失 / 解码 / 显存预检，再启动正式预算。

## 需要实现、核对的代码

| 环节 | 文件 | 待完成事项 |
| --- | --- | --- |
| 原图与几何对齐 | `src/data/letterbox.py`、`src/data/tiktok_dataset.py`、`src/data/iper_dataset.py`、`src/pose_control/qwen21_full/data_runtime.py` | 增加不缩放的原尺寸路径；原图、normal、pose、part、human/valid mask 和 Body25 保持同一坐标。记录原尺寸 / pad 后尺寸 / pad 偏移。 |
| 人脸条件 | `src/pose_control/qwen21_full/face_coordinates.py`、`face_supervision.py`、`data_runtime.py` | Face bbox、landmarks、有效区域与新的 pad 坐标一致；Face 缓存及监督资格按新协议验证。 |
| 原生 Qwen 缓存 | `src/pose_control/qwen21_full/qwen_runtime.py` | 当前 `resolution` 只接受 512/1024，VAE/VLM 输入强制方形。新增原图宽高、源/目标各自的网格和缓存身份；确认模型实际 stride，不沿用方形 `tokens % 4` 假设。 |
| Adapter 与模型 token | `src/pose_control/qwen21_full/qwen_adapter.py`、`controlled_pipeline.py`、`execution.py` | 排查 scalar resolution、sqrt(token_count)、固定 grid 和 pipeline 自动调整尺寸；源/目标不同网格、矩形残差、Face ROI scatter 正确匹配。 |
| 联合损失 | `src/pose_control/qwen21_full/joint_training.py` | 当前 `grid = cache.resolution // 16`、`grid * grid`、square mask / decode 需要改成目标 `(height_grid, width_grid)`；补边不进入 Body/Face 损失。 |
| 训练、推理和签名 | `scripts/train/train_qwen21_joint_pilot.py`、`scripts/eval/eval_qwen21_joint_pilot.py`、`scripts/run/run_qwen21_joint_pilot.py` | 移除旧 512 固定配置；明确 native 预处理签名、形状记录、可恢复的不同分辨率样本顺序。输出按原 B 尺寸裁回并保存，原 B RGB 仍只用于监督 / 离线对照。 |
| 指标与报告 | `src/evaluation/joint_pilot_metrics.py`、`scripts/reports/` | 当前指标处理使用固定 512 口径；新输出与 B 原图关键点保持相同坐标，并明确指标器内部处理及 PCK 归一化。不同协议不混入一个主表。 |

## 验证和查看

已有功能的 CPU 回归检查（不会训练）：

```bash
cd /home/shangguanrz/project/pic-edit
CUDA_VISIBLE_DEVICES= /home/shangguanrz/nihaochen/envs/qwen21/bin/python -m pytest \
  tests/test_qwen21_frozen_bfs_lora.py \
  tests/test_qwen21_joint_training.py \
  tests/test_qwen21_joint_pilot_visuals.py \
  tests/test_qwen21_joint_pilot_summary.py \
  tests/test_qwen21_joint_pilot_resume.py -q
```

原尺寸实现后额外验证：原始像素区域逐像素不变；补边坐标可逆；portrait / landscape / A-B 不同宽高的 token 形状和 loss mask 正确；解码输出裁回 B 原始尺寸；Body / Face 实际梯度有效且基干 / LoRA 冻结；最大片幅真实 GPU 预检不悄悄降采样；同一 J 检查点和匹配的新优化器进行三个分支对照。

本地用浏览器打开 `research/bfs_body_face_joint_pilot_20261008/report/index.html` 查看历史图片。`summary.md` 是未完成旧试验的汇总，会显示尚缺的分支；不能当作三个底座已全部训练完成。Windows 上运行 `scripts/run/sync_qwen21_joint_pilot.ps1` 可再次下载历史实验图片和小文件；权重保留在 sg。

只需在主工作目录继续，不需要再进入 `C:/Users/sg/.codex/worktrees/bfs-body-face-joint/pictureedit`。模型、数据和大型检查点未纳入 Git；原有其他工作区变动仍保留。
