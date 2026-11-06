# Native V5 空间权重实施计划

用户已批准离线生成 GT 的 Body、Joint、Hand 区域和空间权重；先验收 20 张，再构建全量缓存。训练接口可配置，默认保持原 Loss，不启动优化器。

**目标**：复用服务器现有 SAM2.1、DWPose/手部关键点，生成可复用且绑定原图、Canvas 和数据版本的监督侧车，并接入 Native V5。

**架构**：独立离线入口读取 Bilibili 原始 frames 清单。SAM2.1 提供穿衣人体前景，可信关键点生成关节 Gaussian 和手掌/手指软区域。保存三个分量及有效画布；训练按配置融合、限幅、面积下采样，以 [B,N,64] 的 target 网格逐样本归一化。原数据、划分、模型环境和旧缓存保留。

**环境**：Python、PyTorch、NumPy、Pillow、已有 SAM2.1。所有远端新增产物在 /data/shangguanrz/；脚本通过 scp 直传，随后 py_compile。

## 任务与验收

- [x] 数据生成：新增 src/pose_control/qwen21_full/native_v5/spatial_masks.py、scripts/data/build_native_spatial_weights.py、tests/test_native_spatial_masks.py。先测试遮挡/裁切无增强、左右手独立、坐标及面积下采样，再实现。CLI 不下载模型，不重估 SMPL-X，不覆盖已存在文件；每个 GT 与 Canvas 只生成一次。
- [x] 训练接入：新增 spatial_weights.py，修改 training/config.py、data.py、backend.py、objective.py、trainer.py、probes.py、validation.py。SourceSpec 显式绑定空间缓存 index，LossSpec 提供空间系数和上限；拒绝同时叠加旧 face/hand 区域项。零系数走旧路径。测试 Padding 一次、空区域、错位/SHA 失败、逐图归一化、配置序列化和微批梯度。
- [x] 小样本实测：只读核对远端 git 与模型，快照带 UTC 时间。检查 SAM2.1 实际推理，20 张覆盖不同视频及手部状态，保存每张 Body/Joint/Hand、GT 叠加图、权重热图、latent 图和统计。视觉核验通过后才构建全量。
- [x] 全量与交付：缓存 1,960 张有效 GT（若源清单改变则按当前清单重新登记）；每条原记录保留，失败显式报告。全量审计 SHA、数组有限性/范围、Canvas、Padding、目标网格及 lambda=0 基线。更新中文 README，给出真实命令与路径。

## 数值契约

NPZ schema 为 native_spatial_components_v1；数组键 body、joint、hand、valid 均为 Canvas [H,W]，范围 0..1；metadata 为 JSON 字符串，至少包含 schema、gt_rgb_sha256、target_canvas_id、canvas、frame_id 和生成配置。index.json schema 为 native_spatial_index_v1，entries 键为 gt_rgb_sha256 + ':' + target_canvas_id，值含 file:{path,sha256} 和 metadata。几何版本明确记录，沿用当前训练选择 sapiens_v1，不自动替换。

融合在像素画布执行 V * clip(1 + lambda_b*Body + lambda_j*Joint + lambda_h*Hand,1,max_weight)，随后面积下采样到实际 target_token_hw 并按行展开。默认实验系数 .5/.5/1，上限 2.5；每图按 64*sum(weight) 归一化后平均。Padding 不再重复乘 coverage。零系数必须与现有 native_velocity_loss 相等。
