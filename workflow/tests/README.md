# tests/ —— 流水线内部逻辑的单元测试

`workflow` 自身的单元测试集，守护单人与双人两条流水线共用的推理侧共享层 [`../src/pose_edit/`](../../src/pose_edit/README.md)（缓存 / 几何 / DWPose 约定）以及单人线专属的 [`../pose_transfer/`](../pose_transfer/README.md)（PE-I2I 提示词编译）。使用标准 `unittest`，通过 `_workflow_paths.py` 把 workflow 根与仓库根注入 `sys.path` 后，可从 workflow 目录以 `python -m unittest` 运行；用 pytest 时由 `conftest.py` 自动加载。

## 里面有什么

- `_workflow_paths.py`：把 workflow 目录与仓库根加到 `sys.path`。`pytest` 走 `conftest.py`，但 README 文档的 `python -m unittest discover -s tests` 不读 conftest，所以每个测试模块首行 `import _workflow_paths` 自行注入路径。
- `conftest.py`：从 `_workflow_paths` 导入，使 pytest 运行器也能 import `src.pose_edit` 与 `pose_transfer`。
- `test_workflow_common.py`（`StageCacheTests`）：验证 `src.pose_edit/common.py` 的阶段缓存——`make_request` / 指纹 / 产物校验，确保 stage 复用与 sha256 证据正确。
- `test_workflow_geometry.py`（`GeometryTests`）：验证 `src.pose_edit/geometry.py` 的 `project_camera` 针孔投影与无效深度标记、`smplx137_to_dwpose134`、`align_to_source_bbox`、`render_poseguide`、`valid_points`。
- `test_workflow_prompts.py`（`PromptContractTests`）：验证 `pose_transfer/prompts.py` 的提示词契约——`prop_policy` 仅 `source_only`、PE 输出解析（剥离 `<think>` 标签）与编译约束（不得引用 target，`<image3>` 不Swap 源/骨架角色）。
- `test_workflow_cli.py`（`InputFailureTests`）：验证 `../run.py` 的 CLI / 输入失败路径（损坏源图 → 把既有 run 与 latest 标记为 failed 等）；`@unittest.skipUnless(os.name == 'posix')`，仅 Linux 生效。
- `test_dwpose_helper.py`（`BoxIouTests` 等）：`src.pose_edit/dwpose.py` 的契约测试——133→134 关键点约定唯一落点处的 `box_iou` 等纯函数正确性。

## 与上下层的关系

- 守护对象：`../src/pose_edit/`（单人/双人两线共享的推理侧建筑块）与 `../pose_transfer/`（单人线专属提示词编译）。
- 与根 `tests/`（如 `tests/test_deepgen_pose_patch.py` 守护 `patches/`）分工不同：本目录测流水线逻辑，根 `tests/` 测训练/补丁接口。
- 注意：2026-10-13 共享层（缓存/几何/DWPose）已从 `pose_transfer` 迁至 `src/pose_edit`，相关测试同步改 import；提示词编译仍留在 `pose_transfer/prompts.py`，由 `test_workflow_prompts.py` 守护。

## 注意

- 这些是逻辑单测，不拉起 SMPLest-X / Qwen 推理；端到端验证在 `../runs/`。
- 必须在加载 `_workflow_paths` 的上下文运行，否则 import `src.pose_edit` / `pose_transfer` 失败；pytest 由 `conftest.py` 自动处理。
