# add/ —— 外部工具链与第三方数据区（指针）

`add/` **不是本项目源码**，而是外部数据工程与工具链资产区：3DPW 原始数据、conda 离线环境、HumanVid 数据集、YOLO/SAM 权重、数据接入工具、作业日志与迁移记录。主仓库 `.gitignore` **整体忽略 `add/`**。

## 里面有什么（指针式索引）

- `data/`：3DPW 解压原料与 iPER 视频统计（`3DPW/`、`iper_video_stats/`）。
- `datasets/`：数据读取/导出代码与 HumanVid 副本（`dpw_single_pose_edit_dataset.py`、`HumanVid/`）。
- `tools/`：数据接入工具链——HumanVid 处理流水线（`01_*`～`27_*` 阶段脚本）与第三方几何/姿态库副本（SMPLest-X、DWPose 等）。
- `envs/`：各自依赖不同的 conda 环境（`deepgen_pose`、`multihmr`、`sapiens2_hf`、`smpl_clean`、`smplestx_bw`）；`workflow/config.json` 的 `geometry_python`、`smplest_root` 分别指向 `add/envs/smplestx_bw` 与 `add/tools/SMPLest-X`。
- `models/`：三维人体与分割模型（multiHMR、Sapiens 分割、smpl_clean_300、yolo11 系列权重）。
- `outputs/`：数据工程批量产出（3DPW 资产/审计/配对/质检各版本）。
- `logs/`：数据接入作业的远端运行日志（`.log`/`.out.gz`）。
- `migration/`：目录迁移/整理期的映射与搬迁清单。
- `hi4d_contact_semantics_v1/`：Hi4D **接触语义 GT 生成**——把 Hi4D 官方 `smpl/*.npz` 的 `contact>0` 转成可靠的粗粒度身体部位配对 GT（`hi4d_semantic_gt_v1/v2/v3.py` 三版脚本，逐步强化对 0/1-based 索引歧义与分割覆盖的校验），复用 `assets/smpl_vert_segmentation.json`（6890 顶点分割）。产物对接 `outputs/hi4d_contact_semantics_v3_*`、`hi4d_contact_gt_preview*` 等预览批次，服务于 `workflow-multiple/` 的双人接触调研。
- `hi4d_rgb_multihmr_pilot/`：multi-HMR **RGB pilot**——用 `add/tools/multi-hmr` 从 RGB 推断 SMPL-X 并探测接触；含 `pred_contact_probe_v1.py`（基于 SMPL-X 顶点距离阈值绘制接触探针）与 `run_20261014_093532/`（demo_pilot.py / demo_smplx_only.py 及 input/models 子目录）。产物对接 `outputs/hi4d_rgb_multihmr_*`、`hi4d_pred_contact_probe_v1` 预览批次。
- 顶层另见 3DPW 三份原始 ZIP 的符号链接（`camera_tram`/`imageFiles`/`sequenceFiles` → `/data/.../archive/raw/3dpw/`）与若干 YOLO 权重（`yolo11n.pt`、`yolo11x-seg.pt` 等）、pexels URL 清单。

## 与本项目源码的关系

`add/` 是上游依赖与原始资产，源码通过**绝对路径/符号链接**引用它，而非把它纳入版本管理。它与 `deepgen/`、`src/`、`workflow/` 是"被消费"的外部底座，而非项目自有逻辑。例如 `workflow` 的姿态估计完全依赖 `add/envs/smplestx_bw` + `add/tools/SMPLest-X`；`patches/` 不直接涉及 `add/`。

## 注意与坑

- **外部资产区，体量大且多为不可再生原始数据，不要随意删除或改动。**
- 原始 3DPW ZIP 已冷归档到 `/data/shangguanrz/pic-edit-archive/cleanup_20261003/raw/3dpw/`，原位置仅留链接。
- 本区不在 Git 内；若要把自有脚本纳入版本控制，需单独审查 `.gitignore` 忽略规则。
