# archive/add_picture_edit_rescue_20260921/ —— add_picture_edit 工作区救援快照

2026-09-21 对外部工作区 `add_picture_edit` 做的**救援快照**：当时该工作区（外部 `add/` 区下关于「加图编辑」的脚本与数据）处于易丢失/易误改状态，先将其冒烟运行结果冻结留存，作为事后核对与恢复的基线。

## 里面有什么

- **`smoke_test_outputs/`** —— 救援当日的冒烟运行日志，三个 `.log` 均时间戳 2026-09-21 10:53：
  - `t2i_official.log` —— 文生图（text-to-image）官方流程冒烟日志。
  - `edit_official.log` —— 编辑（edit）官方流程冒烟日志。
  - `smoke_test.log` —— 通用冒烟测试日志。
  - 另有脚本自动生成的占位 `README.md`。

## 与上下层的关系

- 对应 live 侧的外部 `add/` 工作区（3DPW 单/双人姿态编辑相关脚本，参见 `cleanup_20261003/add_backups/` 中 `dpw_single_pose_edit_dataset` 系列）。
- 它比 `cleanup_20261003`（2026-10-03）早约两周，是更早一次针对 `add/` 区的保护动作；后者才是系统性的目录整理。

## 注意与坑

- 只读救援副本。日志仅记录冒烟是否跑通，不含完整产物；需要复现时应回到 `add/` 工作区的对应脚本与数据。
- 不要把这些日志当作实验结果证据，它们只是「当时流程可运行」的快照。
