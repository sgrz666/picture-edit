# tests：双人线测试套件

双人线测试套件，对应 ../README.md「tests/ 11 个模块全绿」。用 unittest 编写，docstring 写的是**契约**而非实现——专防"跑一遍才发现"的静默错误。

## 里面有什么

共享文件：`conftest.py`（pytest 入口）；`_workflow_multiple_paths.py`（把 `workflow-multiple/` 与仓库根加进 sys.path，unittest discover 不读 conftest 故每模块先 import 它）。

11 个测试模块（与 README「11 个模块全绿」一致）：
- `test_dual_geometry.py` —— 几何/定标与双人槽位。
- `test_dual_fitting.py` —— fit 向导关节映射须与 fit 报告一致。
- `test_fit_gate.py` —— 拟合门判定项须是投影几何误差（`hip_gap_error_norm_gt`）。
- `test_dual_evaluation.py` —— `region_boxes` 逐人区域检测（整图并框时）。
- `test_dual_prompts.py` —— 生成器只指向收到的图，禁引用 B。
- `test_compare_dual.py` —— 成对比较只比同一对，声明无法比的情况。
- `test_appearance.py` —— 外观度量（裁剪钳制、DINOv2 余弦）。
- `test_matrix_audit.py` —— 覆盖从产物算，成对行只保一变量。
- `test_run_names.py` —— run 名契约，防 64 字符上限静默丢格（已放宽 128）。
- `test_size_compliance.py` —— 出图尺寸合规分组/比值。
- `test_torso_ratio.py` —— 量具本身：mask 框与 torso÷框高 拒绝缺值。

## 与上下层的关系

- 被测代码在 `../`（双人线流水线模块）与共享包 [`../../src/pose_edit/`](../../src/pose_edit/README.md)。
- 单人线对照 `../../workflow/README.md`。

## 注意与坑

- 跑法：仓库根 `pytest`，或 `python -m unittest discover -s tests`（靠 `_workflow_multiple_paths` 注入路径）。
- `test_run_names` 从 `../pairs/`、`../pairs_cross/` 实目录派生名字，加长对名会提前触发，非硬编码。
- 契约锁：改几何/拟合/门禁后必须重跑；改共享包 `src/pose_edit` 指纹会让批量 PE 阶段失效。
