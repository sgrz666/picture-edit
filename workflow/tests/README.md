# tests/ —— 流水线内部逻辑的单元测试

`workflow` 自身的单元测试集，覆盖 `pose_transfer` 的缓存、几何、提示词契约与 CLI 输入失败路径。使用标准 `unittest`，通过 `conftest.py` 把 workflow 根加入 `sys.path` 后，可从仓库根以 `python -m unittest` 运行。

## 里面有什么

- `conftest.py`：把 workflow 根目录注入 `sys.path`，使测试在仓库根也能 import `pose_transfer`。
- `test_common.py`（`StageCacheTests`）：验证 `common.py` 的阶段缓存——`make_request` / 指纹 / 产物校验，确保 stage 复用与 sha256 证据正确。
- `test_geometry.py`（`GeometryTests`）：验证 `geometry.py` 的 `project_camera` 针孔投影与无效深度标记、`smplx137_to_dwpose134`、`align_to_source_bbox`、`render_poseguide`、`valid_points`。
- `test_prompts.py`（`PromptContractTests`）：验证 `prompts.py` 的提示词契约——`prop_policy` 仅 `source_only`、PE 输出解析与编译约束（不得引用 target）。
- `test_cli.py`（`InputFailureTests`）：验证 CLI / 输入失败路径（图片缺失、非单人输入等错误处理）。

## 与上下层的关系

- 守护对象：`../pose_transfer/` 的代码正确性。
- 与根 `tests/`（如 `tests/test_deepgen_pose_patch.py` 守护 `patches/`）分工不同：本目录测流水线逻辑，根 `tests/` 测训练/补丁接口。

## 注意

- 这些是逻辑单测，不拉起 SMPLest-X / Qwen 推理；端到端验证在 `../runs/`。
- 必须在加载 `conftest` 的上下文运行，否则 import `pose_transfer` 失败。
