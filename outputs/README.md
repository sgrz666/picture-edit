# outputs/ —— 临时运行输出目录

项目级临时运行输出目录。与 `workflow/runs/`（结构化、带 sha256 证据链的分运行产物）不同，这里偏向一次性/中间产物的暂存。

## 里面有什么

项目级临时输出，内容随调用方而定，没有固定契约。当前（2026-10-14 起）存放的是一批 **Hi4D 双人接触/外观预览**产物，来自接触语义与 multi-HMR/SMPL-X 预览脚本的一次性运行：

- `hi4d_contact_gt_preview/`、`hi4d_contact_gt_preview_fixed/`、`hi4d_contact_mesh_preview/`、`hi4d_contact_semantics_v3_20261014_092437/`：双人接触区域的真值/网格/语义预览图。
- `hi4d_pred_contact_probe_v1/`：预测接触探针预览。
- `hi4d_rgb_multihmr_20261014_093532/`、`hi4d_rgb_multihmr_smplx_20261014_093730/`：multi-HMR 与 SMPL-X 两种估计在 Hi4D RGB 上的对照预览。

除此之外本目录基本为空；这些预览批次与 `workflow-multiple/` 双人线的接触/外观调研相关，但**不属于**任何带证据链的正式 run。

## 与上下层的关系

- 互补：`workflow/runs/` 才是流水线正式、可追溯的产物落脚地（见 `workflow/runs/README.md`）。
- 若某脚本需要落盘临时中间结果，可能写到这里；使用方应自行记录路径。

## 注意与坑

- 临时目录，内容可能随时被覆盖或清理，**不要依赖其持久性**。
- 想长期保留某次运行结果，应放在 `workflow/runs/<run>/` 或 `results/` 下，而非此处。
