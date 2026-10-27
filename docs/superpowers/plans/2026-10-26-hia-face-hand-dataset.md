# HIA 脸手清晰数据集实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. 独立模块并行实施，主任务负责集成、真实样本检查和运行验收。

**Goal:** 用 HIA-RAW 构建可追溯、脸与双手清晰、无视频交叉的 train/val/test。

**Architecture:** 来源解码、局部质量和划分导出分开实现。CPU ONNX 复用既有权重，先小批计时和视觉检查，再按明确范围扩展。

**Tech Stack:** Python、OpenCV、NumPy、ONNX Runtime、现有 DWPose/ArcFace、ZIP、JSONL、PNG。

## 任务与责任

- [ ] source worker：scripts/hia_source.py、tests/test_source.py。全库清点，稳定跨分卷抽样，内存视频解码，正确 RGB 索引/PTS，SHA256；测试采样确定性、索引去重、ZIP成员不串序列。
- [ ] quality worker：scripts/hia_quality.py、tests/test_quality.py。wholebody134 可信 ROI、逐侧几何门、原生 ROI 质量、明确拒绝原因；用合成清晰/模糊图和错误手窗口验证，不改现有模型模块。
- [ ] split worker：scripts/hia_finalize.py、tests/test_finalize.py。视频原子分组、CHAMP冻结表、内容重复合组、train/val/test、同镜头pair、完整校验与JSONL导出；测试冻结归属冲突、重复哈希隔离、pair不跨split。
- [ ] 主任务：scripts/hia_build.py、README.md。参数化入口、可恢复日志、小批真实数据、原生ROI复核、性能计时、全量来源清点及扩大候选、身份近邻检查、最终复核和交付。

## 共享接口

来源视频行包含 clip_id、shard、zip_path、member、source_key；帧行包含 clip_id、frame_idx、timestamp_sec、shot_id、source_video_sha256、original_hw。RGB按BGR ndarray交给质量模块。关键点用 candidate[134,2] 与 confidence[134] 的像素坐标。

质量函数 analyze_frame(image, candidate, confidence, n_persons, thresholds=None) 返回 JSON 可序列化字典：strict_pass、reject_reasons、regions（face/hand_a/hand_b，带bbox_xyxy、geometry_pass、metrics、passed）、body。缺失/无效值一律拒绝，不能缺字段默认通过。

最终帧行额外包含 frame_id、rgb_path、rgb_sha256、pose_path、quality、candidate、confidence。finalize 接收已选帧列表、输出目录与seed，返回summary；不修改媒体。先分组后同组配对，明确数字ID不等于身份ID。

## 运行与验收顺序

1. 三模块分别先跑有意义的失败测试，再实现并运行通过。
2. 本地 py_compile；scp 代码至服务器新实验目录，服务器 py_compile。
3. 清点25,000源视频，跨25分卷小批抽样，测实际每帧耗时与通过率。
4. 生成原生脸/手复核页，检查高分模糊、贴错手腕、背景掩盖等反例。阈值配置版本化，记录尚无人工标签。
5. 按性能扩大分层视频样本，覆盖25分卷，避免资源争抢。明确记录processed/total，不以抽样冒充整库处理。
6. 身份/重复来源合组，冻结划分，构建pair与JSONL；运行verify全量路径/哈希/分组验证。
7. 补充中文README、运行统计、复核覆盖，向用户给绝对路径、各split数量和可复制验证/读取命令。
