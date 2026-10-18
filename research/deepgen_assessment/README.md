# research/deepgen_assessment/ —— DeepGen-1.0-diffusers 纯基座生成能力评估

本目录评估 **DeepGen-1.0-diffusers（2.47B MMDiT，frozen）纯基座**在不同提示词与采样配置下的原始渲染能力——即不加任何自研 adapter 时基座本身的图文一致性上限。它是技术线 1 的「能力基线」参照，配合 `research/legacy_202609/` 的适配性评估 docx 一起看。所有产物是脚本 + 渲染图 + PDF 报告，不含训练代码。

## 里面有什么

| 文件 / 子目录 | 内容 |
|---|---|
| `build_report.py` | 渲染报告构建脚本（27 KB，顶层定义 `p`/`h`/`page`/`table` 等，生成可离线查看的页面） |
| `deepgen_pipeline_audited.py.txt` | 被审计/带注释的 DeepGen 推理管线源码（63 KB，`"""` 文档串开头，作可复现渲染入口） |
| `render/` | 第一轮纯基座渲染样本（15 文件，含 `DeepGen人物姿态图像编辑适配性评估.pdf` 与 page-*.png） |
| `render_final/` | 收尾轮渲染样本（10 文件，PDF + page-*.png） |
| `render_v2/` | 第二轮渲染样本（10 文件，PDF + page-*.png） |

## 与上下层的关系

- 评估对象：技术线 1 的基座 `src/pose_control/v6/` 所依赖的 DeepGen-1.0-diffusers。
- 配套结论：`research/legacy_202609/DeepGen人物姿态图像编辑适配性评估.docx`（同主题适配性评估正文）。
- 价值定位：为「VAE 8× 下采样是手/脸细节硬瓶颈」「1024 分辨率是单一最有效杠杆」等结论提供基座侧证据（结论以 `docs/架构审查_20261012.md` 与训练横评为准，本目录只给基座渲染）。

## 注意与坑

- `deepgen_pipeline_audited.py.txt` 是**对照用带注释副本**（扩展名 `.txt` 防止被当作运行模块），真实推理管线在主树 `src/`；两者可能不同步。
- 三个 `render*/` 是不同时间/配置的渲染批次，比较时先看各自 PDF 内的配置说明，别跨批次混比。
- 本目录只测**基座生成**，不含身份保持/姿态控制指标；控姿效果请转 `research/bfs_transfer_audit_20261004/` 与 `docs/experiments/`。
