# runs/ —— 两阶段流水线的运行产物与证据链

本目录存放每次 `workflow/run.sh` / `batch.sh` 执行的全部产物。每个子目录是一次独立运行，内部按阶段分子目录：`inputs/`、`geometry/`（SMPLest-X → DWPose-134）、`prompt/`（PE-I2I 角色编译）、`generation/`（Qwen-2.1 基座只读 A + 骨架）；schema v2 起还带 `evaluation/`（生成图回估评分）。根 `run.json` 记录 schema、settings、各阶段 `cache_hit`、outputs 的 sha256 证据链与 `verification` 结论；`outputs.json` 是路径速查表。

## 里面有什么（分两代；实时清单与数量以 `report.py` 为准）

**整改前的 legacy 运行（schema v1，6 个）**——记录格式已变，不再可 `--verify`，`report.py` 会标为 `legacy`；是整改前的冻结证据：

- `demo`：最早 1024 全缓存复用的端到端样例，三步 `cache_hit=true`，`status=complete`。基线链路跑通。
- `testset_hard05`：hard_05 配对 1024，仅 geometry 完成即中断（`active_stage=prompt`），无最终图。
- `testset_hard05_768`：同配对降到 768，三步全新运行（`cache_hit=false`），complete。768 基线。
- `testset_hard05_fitgeo` 与 `testset_hard05_promptv2`：同一对图、同一几何的两个**手工提示词变体**——几何与 PE 阶段记录逐字节相同，两者只差提示词文本（`run.json` 亦相同，曾一度被误读为"拷贝残留"）。手写提示词来源哈希 `17b9c608…`。
- `testset_medium02_768`：medium_02 配对 768 基线，complete。

**整改后的正式运行（schema v2，9 个）**——全测试集 7 对在 768 / seed 42 / 30 步 / PE 提示词下的实况，附 `evaluation/` 回估评分：

- `hard_01`~`hard_05`、`medium_01`、`medium_02`：逐对基线运行。其中 `hard_01`、`medium_01` 的 A 图有多人（3 人 / 2 人），用 `--subject largest` 复跑；`hard_05` 是全场最难的一对（`source_person` 定标下 body PCK 仅 0.056）。
- `hard_05_torso`：hard_05 的躯干定标变体（`--framing source_torso`，尺度 0.410→0.977，躯干错配 2.35×→1.02×，body PCK 0.056→0.389）。
- `hard_05_manual`：同一躯干定标下换手工提示词（`--prompt-file`），body PCK 进一步到 0.667；`latest_run.json` 当前指向它。

逐对指标表（body/feet/face/hand 分组 MPJPE、PCK@0.2/0.5）以 [`../README.md`](../README.md) 的「实况」一节与 `report.py` 输出为准，本目录不复制。

**2026-10-13~14 之后新增的运行（增补）**——上面两代共 15 个 run 现已整体作为批处理目录收在 `runs/test/` 下（物理路径 `runs/test/<name>`，含 `medium_01` 嵌在 `runs/test/demo/medium_01/`）；`report.py` 递归仍能索引到全部 19 个（7 对 × 多次 + legacy）。以下为本次增补时实际 `ls` 到的、未包含在上述两代里的运行：

- `runs/test/q8l_hard_02`、`runs/test/q8l_hard_05`、`runs/test/q8s_hard_02`、`runs/test/q8s_hard_05`：Q8 量化基座（GGUF）对照实验，large/small 两档（q8l/q8s）分别配 hard_02、hard_05 两对；参数沿用基线（768 / 30 / seed 42），用于与全精度基座比动作迁移质量。量化档与具体模型以各 run 的 `generation.json` 为准（待补充确认）。
- `scale_anchor_20261013/`：source-anchor 生成消融实验产物（对应 [`../diagnostics/2026-10-13-scale-generation/`](../diagnostics/README.md)），用 A 实测尺度 + 骨盆锚定重生成 4 坏例 + `medium_02` 对照 + `hard_02` 提示词修正版；`experiment.json` 区分其为消融而非端到端 PE 流程。
- `wp1_accept_hard02/`：WP1 验收对照 run（hard_02 配对，带 `evaluation/`），尾部 `WP1_ACCEPT_DONE`、body PCK@0.2=0.944。
- `torso_check_person_20261014/`：source_person 默认定标回归对照（同验收 hard_02 输入，1024 画布），验证检出框数值与已验收 run 一致（仅 `inputs/`+`geometry/`，无 generation）。
- `torso_measured_hard02_20261014/`：source_torso 改用 A 实测躯干后的真实重跑（hard_02 输入，1024），`VERIFICATION validated`，是 [`../README.md`](../README.md)「source_torso 改用实测 A 躯干」节的实证 run。
- `tiktok/`：TikTok 舞蹈帧单人测试集（8 对，竖屏 604×1080），含 `pairs/`、`candidates_v2.json`、`tt_*_` 8 个完整 run、`results.md`/`summary_grid.png`、`panels/`（软链到 [`../scripts/six-picture-create/output/tiktok/`](../scripts/README.md)）；自带 `README.md`。
- `PE-GGUF/`：当前为空目录（占位），预留给 PE 的 GGUF 量化模型 / 相关运行，暂无产物。

## 与上下层的关系

- 上游：`../run.py`（入口 `../run.sh`，批量入口 `../batch.sh`）依据 `../config.json` 与命令行生成；`../patches/` 是对 SMPLest-X 的本地补丁（几何阶段加载）。
- 下游消费：`../evaluate.py`（回估生成图并对照请求骨架打分）、`../report.py`（汇总逐对指标）；运行日志在 `../logs/`（`batch_testset_768_20261012.log`、`hard05_torso_20261012.log` 等）。
- 指纹与复验逻辑见 [`../src/pose_edit/`](../../src/pose_edit/README.md)（缓存指纹在 `common.py`、几何/估计器/DWPose 等共享层，2026-10-13 由 `workflow/pose_transfer/` 迁来）；基座流水线总说明见 [`../README.md`](../README.md)。

## 注意与坑

- **代码状态分三代**（这三条哈希来自 2026-10-13 共享层迁移**前**的 run，其 `geometry/stage.json` 仍记旧路径 `pose_transfer/geometry.py`；迁移后的 run 记 `src/pose_edit/geometry.py`，见 [`../src/pose_edit/README.md`](../../src/pose_edit/README.md)）：`6603a819…`（首轮批处理 hard_02/03/04/05、medium_02）、`6a57ff88…`（hard_01、medium_01 的 `--subject largest` 复跑）、`5ac98986…`（两个 hard_05 变体，含 `--framing`）。对旧 run 执行 `--verify` 会报 `workflow code changed`——产物完好，只是代码已演进；全测试集尚未在同一份代码状态下重跑过。
- `cache_hit=true` 直接复用既有 stage 不重算；改设置后强制重跑用 `--force`，建议新建 run 名保留旧产物。
- 批量实验用 `--pair $PAIRS/<name>`，不依赖 `../pictures/`。`$PAIRS` 在 sg 是 `../experiments/2.qwen-test/qwen-2.1-test/data/testset/pairs`（5 族分组），Windows 侧少一层 `2.qwen-test/`；`batch.sh` 与 `report.py` 两种都会自己试。
- **目录深度有三种**：`runs/<run>`、`runs/test/<run>`、`runs/<批次>/<run>`，还有 `medium_01` 那样嵌在另一个 run 里的（`runs/test/demo/medium_01/`）。2026-10-14 起 `report.py` 按 `run.json` 递归发现，早先按一层遍历的版本会漏 run、也会在批次目录上崩。
- 整改前的整树备份在 `/data/shangguanrz/pic-edit-archive/workflow_runs_backup_20261012/`（含 `SHA256SUMS`）；审查与整改全过程见 `docs/workflow审查与整改_20261012.md`。
- **`runs/test/` 是整批迁移后的归宿**：上面两代 15 个 run 及 q8l/q8s 现都在 `runs/test/` 下；`medium_01` 实际嵌在 `runs/test/demo/medium_01/`，按一层遍历会漏掉它、`report.py` 已修复能递归索引到。legacy v1 六个（`demo`、`testset_hard05*`、`testset_medium02_768`）随整批迁到 `runs/test/`，schema 已变、`report.py` 标 `legacy` 不可再 `--verify`；旧文档 / 日志里写的 `runs/<name>` 顶层路径是迁移前位置。
- 2026-10-16 新增的 `../lowlatency_probe.py` 低延迟探测线（llama.cpp / GGUF，三方同输入对比 HF PE / GGUF PE / GGUF 4B，见 [`../README.md`](../README.md)「低延迟探测（llama.cpp / GGUF）」一节）产出落在调用方 `--out` 指定的**实验目录，不在 `runs/` 下**；`runs/PE-GGUF/` 仍为占位空目录（10-16 建），预留给 PE 的 GGUF 量化模型 / 相关运行，暂无产物。
