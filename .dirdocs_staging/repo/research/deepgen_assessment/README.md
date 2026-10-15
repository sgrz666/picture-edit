# research/deepgen_assessment/ —— DeepGen-1.0 适配性评估报告：docx 生成器与三代 PDF 渲染快照

> 工作区路径：`/home/shangguanrz/project/pic-edit/research/deepgen_assessment`　|　本级：2 个实证文件 + 3 个子目录　|　子树规模：17 MB（du 实测），32 张页面 PNG + 3 份 PDF，统计时间 2026-10-12

## 一、这一层在干什么

2026-09-17 的问题：**DeepGen-1.0-diffusers 能否作为人物姿态图像编辑的第三阶段基座**（当时约束：生图主干 ≤6B、单卡 RTX 6000D 84GB）。产出了一份技术评估报告（docx）及其三代 PDF 渲染。文字版结论与显存预算的完整论证在姊妹目录 `../legacy_202609/第三阶段图像编辑调研与显存预算.md`；本目录是报告的**排版定稿过程快照**，不是模型渲染样本——`render*` 里是报告逐页 PNG，不是生成图像。

## 二、层级地图

| 子目录 | 装什么 | 内容实证 | 与本级其他子目录的关系 |
|---|---|---|---|
| `render/` | 报告初版渲染 | 14 页 PNG + 同名 PDF（567,201 B） | 初版：14 页版式 |
| `render_v2/` | 收紧为 9 页的第二版 | 9 页 PNG + PDF（566,278 B） | 版式重排后 14 页缩为 9 页 |
| `render_final/` | 定稿版 | 9 页 PNG + PDF（566,105 B） | 与 v2 相比仅 page-1、page-8 逐字节相同（md5 实测），page-2~7、page-9 均有微调 |

- 顶层 `build_report.py`（27 KB）：用 python-docx 程序化生成 `DeepGen人物姿态图像编辑适配性评估.docx`；脚本声明输出到其父目录（`research/`），实际档案现存于 `../legacy_202609/`（52 KB，2026-09-17）；含页眉页脚、样式与表格构造。
- `deepgen_pipeline_audited.py.txt`（63 KB）：DeepGen-1.0-diffusers 官方 standalone pipeline 源码的带注释审计副本，用于核实"保留参考图像路径 + 残差注入接口"这一采纳依据。

## 三、关键文件与实证内容（报告结论本身）

- 评估结论（docx 首段实测摘录）：**建议将 DeepGen 纳入第三阶段优先原型基座**，冻结主干 + 100M~300M 可训练结构适配器；风险是双人接触与外观保持未验证，且"用 Qwen 编码器 ≠ 属于 Qwen-Image-Edit 家族"，若验收限定 Qwen-Edit 家族则不能替代交付。
- 权重口径：DiT 2,469,663,936 + VLM 3,754,622,976 + SCB connector 432,449,532 + VAE 83,819,683，合计 6,740,556,127 元素（与 `../legacy_202609/checkpoint_audit.json` 逐项一致；revision `85c2ed25…`，BF16 等效 12.555 GiB）。
- 后续补充（见 legacy md）：按"Qwen-Edit 家族压缩模型"收紧选型后，**DeepGen 主路线推荐被暂否**，改选 Qwen-Image-2.1 线——这是本目录被归档为"证据层"而不再更新的原因。

## 四、注意事项 / 调用路径

- 三代渲染是**同一份报告的版式迭代**，内容同一；看正文用 docx（`../legacy_202609/`），比排版差异用 `render_final/`。
- `build_report.py` 依赖 python-docx 与系统 PDF 转换，输出路径写死为父目录，重跑会覆盖 legacy docx，慎动。
- 采纳/否决结论与数字证据落点：`../legacy_202609/checkpoint_audit.json`（权重规模）、`../legacy_202609/第三阶段图像编辑调研与显存预算.md`（预算与选型修正）。
