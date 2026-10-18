# panels/ —— 双人线对比面板图工具

`panels/` 是双人线的拼图工具：把一次双人 run 的关键产物（A/B 原图、合并骨架、最终生成图等）拼成带标注的对比面板图，用于人工审图与报告插图。与单人线的 [`workflow/scripts/six-picture-create/`](../../../workflow/scripts/README.md) 是同级工具，只是面向双人场景。

## 里面有什么

- `code/panel_dual.py` —— 工具源码：读 run 目录产物，按双人场景拼面板（骨架图分人绘制、合并图不跨人连线）。
- `index.json` —— 已生成面板的索引（面板路径 + 对应 run）。⚠️ 其中记录的面板路径写作 `workflow-multiple/panels/...`，是失效旧值——面板实际就存在本目录下（见下）。
- 顶层 `*.png` —— 已生成的对比面板，按 run 命名（如 `hug_pair00_084_c4_r000.skeleton_only.b_canvas.png`、`backhug_pair02_037_c64_r545_b_canvas.png`）。
- `eight-pictures/` —— 八图变体面板目录（当前为空，疑似脚本创建的暂存位，待补充）。

## 与上下层的关系

- 上游：读 [`../../runs/`](../../runs/README.md)（及 `../pairs/`）里各 run 的产物拼图；脚本隶属 [`../`](../README.md)（双人线私有工具层）。
- 下游：面板图供人工审图与 [`../../README.md`](../../README.md)「实况」节的报告引用。
- 用法与坑见 [`../README.md`](../README.md)（scripts/ 层说明）；双人线总说明见 [`../../README.md`](../../README.md)。

## 注意与坑

- 面板是**派生物**：run 产物更新后需重跑工具刷新，面板图本身不进证据链（证据以各 run 的 `run.json`/`stage.json` 为准）。
- `index.json` 路径失效问题如上；引用面板时以磁盘实际文件为准，别信索引里的旧路径。
- 本目录结构未严格遵循 `script/`+`output/`+`README.md` 三件套（代码在 `code/`、图在顶层），新工具请按标准样板建。
