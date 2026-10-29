# Qwen-Image-2.1-Fun-ControlNet-Union 实验汇报（Word）

## 这个目录是什么

给汇报用的一份 Word 文档，把服务器上 **qwen-union-controlnet（Qwen-Image-2.1-Fun-Controlnet-Union）** 相关的全部测试实验
整理成一个文件：每条实验线写清「什么实验 / 实验条件 / 实验输入 / 出图结果 / 量化指标 / 结论」，并把生成图片尽量放全。

- 主交付件：`Qwen-Image-2.1-Fun-ControlNet-Union_实验汇报_20261027.docx`（53 页，横向 A4，149 处图片，8.4 MB）
- 同名 `.pdf` 是 LibreOffice 转出来的版面核对件，内容与 docx 一致；要改文字就改 docx 或改脚本重新生成。

## 里面有什么

| 路径 | 是什么 |
|---|---|
| `Qwen-..._实验汇报_20261027.docx` | 汇报正文 |
| `Qwen-..._实验汇报_20261027.pdf` | 同一份内容的 PDF 预览件 |
| `build_report.py` | 生成脚本。改内容优先改这个再重跑，不要手改 docx |
| `assets/` | 从 sg 只读复制下来的 170 张原始 PNG（= 131 个不同内容），按实验线分目录 |
| `meta/` | hard_05 六臂实验的 prompt、sample_manifest、execution_summary、prep_manifest |
| `_docimg/` | 脚本生成的缩放 JPEG 中间件，可随时删，重跑会重建 |

## 收录了哪四条实验线

1. **方案 A** `experiments/6.qwen-controlnet union/qwen21_union_migration_v1`
   —— 官方 Union 原生迁移 + 三关验证；48 张「6 样本 × 4 臂 × 2 seed」矩阵全部收录，另有第 1 关的
   Pose/Depth/Inpaint 出图、控制响应面板、消融探针、8 步预检（附录 B）。
2. **hard_05 六臂** `workflow/experiment/qwen-union`
   —— 3 种骨架布局 × 2 种生成模式；6 张出图 + 13 张控制条件图 + 九宫格对比。
3. **方案 B** `experiments/6.qwen-controlnet union/qwen21_union_dataset_adaptation_v1`
   —— 多数据集 LoRA 训练；只有 4 张两步 smoke 出图与 8 张数据 QA 面板（正式 1024 张 test 未启动）。
4. **Pro Adapter** `experiments/6.qwen-controlnet union/qwen21_pro_adapter_v1`
   —— 人物任务分支；22 张出图（step0/step200 配对、val 对照、seed 消融、CFG 消融）。

## 关键数字（正文里都有出处）

- 方案 A：单人严格 PCK@0.2 随控制强度单调上升，`hard_02` 0.750 → 0.857 → 0.964；代价是 `union_s10` 身份分
  0.916 → 0.585，且骨架线被画进画面。双人检出人数从 1.5/1.0/3.5 掉到 1.0。判定：**需要进一步输入适配**。
- hard_05 六臂：Union 把 Body PCK@0.2 从 Native Qwen 的 0.3333 提到 T2a 0.8889 / T4b 0.6111；
  但 ArcFace 身份只有 0.07~0.24，Native 多图注入是 0.8259。
- 方案 B：overfit200 固定平均 loss 只降 **17.7868%**（门槛 20%）→ `gate_pass=false`，后续阶段未启动。
- Pro Adapter：200 步全局 MSE 只降 **5.67%**；25 步五组消融 -1.61% ~ -1.85%，全部远低于门槛。

## 注意与坑

- **图片是受保护资源**：`assets/` 全部是 `cp -n` 只读复制来的副本，服务器原件未做任何修改、移动或删除。
  重新取数走 `/tmp/union_report_stage_20261027/` 那套 staging 脚本，不要直接在实验目录里动。
- **native 与 union_s0 逐字节相同**：170 个文件里 39 个是重复内容（md5 已核）。文档里两个格子放同一张图是事实，
  不是漏图；Word 内部也做了去重，所以 media 数（135）小于图片格子数（149）。
- **方案 B / Pro Adapter 的出图不是效果证据**：前者是 2 步 smoke，后者标注 `engineering_probe`
  （`quality_claim: false`）。文档里已就近写明，汇报时不要把它们当评测结果讲。
- **不属于本族但容易被误收**：`/data/.../qwen21_full_native_v1`（自述「无需 Union 模型」，归 3.singel man）、
  `hia_dataset_v2_20261026`、`hia_face_hand_dataset_20261026`（数据集构建线，无 Union 生成图）、
  `union-adapter/fulladapter`（只读代码快照，不可独立运行、无出图）。这四处在本次审计里都确认过并排除。
- 表格里 `\n` 由 python-docx 转成 `<w:br/>`，在 Word 里是单元格内换行；改脚本时不要改成两个 run。
- 重跑：`python build_report.py`（依赖 python-docx + Pillow，本机已装）。
