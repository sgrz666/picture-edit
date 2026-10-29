# reports/ —— 成果报告与交付物导航

`reports/` 存放面向技术评审、课题组汇报与成果交付的正式交付物，包括 Word/PDF 报告、多栏对比面板（Panel）与交互式可视化页面。

---

## 一、报告目录导航总表

| 目录 / 文件 | 报告主题与内容 | 关联实验 / 工具 | 交付形态 |
|---|---|---|---|
| [`native_v5_fixedpair_20261026_v1/`](native_v5_fixedpair_20261026_v1/README.md) | **V5 固定样本实验全景报告**<br>包含 TikTok (0374/0126) 与 Hi4D (0082/0112) 每 30 步过程演变、同输入消融对照与局部 ROI 细节 | `tiktok-fixedpair-v5_20261025_v1`<br>`hi4d-fixedpair-v5_20261025_v1`<br>[`scripts/reports/`](../scripts/reports/README.md) | 单文件 HTML、PDF、`manifest.json` 与原始图片证据包 |
| `qwen_union_controlnet_20261027/` | **Qwen ControlNet Union 控制能力报告**<br>包含 TikTok 8 对样本横向实测、4 臂对比与姿态跟随/身份保持边界分析 | `workflow/runs/tiktok/union-test/`<br>`6.qwen-controlnet union` | 指标表格、对比图板与评估 JSON |
| `architecture/` | **系统架构设计图册**<br>包含端到端多流网络架构图、可编辑 Draw.io、PNG/PDF 与实现证据 | 全仓架构设计 | 矢量图与渲染图 |
| `evaluation/` | **TikTok 与 Hi4D 评测指标库**<br>包含定标测绘、ROI 区域裁切与客观指标明细 | `scripts/eval/` | JSON 指标与对比切片 |
| `legacy_demo_202609/` | **早期阶段性里程碑成果存档**<br>早期 MMDiT 适配器推理图板与演示包 | 早期成果 | 图片与说明 |

---

## 二、报告构建与生成工具链

所有报告的生成均基于只读审计，不修改已有实验权重与图片：
- 导出只读证据包：[`scripts/reports/export_qwen21_fixedpair_report.py`](../scripts/reports/export_qwen21_fixedpair_report.py)
- 构建单文件 HTML 视觉报告：[`scripts/reports/build_qwen21_fixedpair_visual_report.py`](../scripts/reports/build_qwen21_fixedpair_visual_report.py)
- 离线无头浏览器导出 PDF：[`scripts/reports/render_qwen21_fixedpair_report.cjs`](../scripts/reports/render_qwen21_fixedpair_report.cjs)

---

## 三、远端外部数据盘产物落点

大体积高分辨率证据图与全量评测切片持久化于数据盘：
- TikTok 报告落点：`/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/tiktok-fixedpair-v5_20261025_v1/report/`
- Hi4D 报告落点：`/data/shangguanrz/pic-edit-experiments/qwen21_hi4d_dual/hi4d-fixedpair-v5_20261025_v1/report/`
