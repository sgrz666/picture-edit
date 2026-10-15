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

## 与上下层的关系

- 上游：`../run.py`（入口 `../run.sh`，批量入口 `../batch.sh`）依据 `../config.json` 与命令行生成；`../patches/` 是对 SMPLest-X 的本地补丁（几何阶段加载）。
- 下游消费：`../evaluate.py`（回估生成图并对照请求骨架打分）、`../report.py`（汇总逐对指标）；运行日志在 `../logs/`（`batch_testset_768_20261012.log`、`hard05_torso_20261012.log` 等）。
- 指纹与复验逻辑见 `../pose_transfer/`；基座流水线总说明见 [`../README.md`](../README.md)。

## 注意与坑

- **代码状态分三代**（按各 run `geometry/stage.json` 所记 `pose_transfer/geometry.py` 哈希区分）：`6603a819…`（首轮批处理 hard_02/03/04/05、medium_02）、`6a57ff88…`（hard_01、medium_01 的 `--subject largest` 复跑）、`5ac98986…`（两个 hard_05 变体，含 `--framing`）。对旧 run 执行 `--verify` 会报 `workflow code changed`——产物完好，只是代码已演进；全测试集尚未在同一份代码状态下重跑过。
- `cache_hit=true` 直接复用既有 stage 不重算；改设置后强制重跑用 `--force`，建议新建 run 名保留旧产物。
- 批量实验用 `--pair $PAIRS/<name>`，不依赖 `../pictures/`。`$PAIRS` 在 sg 是 `../experiments/2.qwen-test/qwen-2.1-test/data/testset/pairs`（5 族分组），Windows 侧少一层 `2.qwen-test/`；`batch.sh` 与 `report.py` 两种都会自己试。
- **目录深度有三种**：`runs/<run>`、`runs/test/<run>`、`runs/<批次>/<run>`，还有 `medium_01` 那样嵌在另一个 run 里的（`runs/test/demo/medium_01/`）。2026-10-14 起 `report.py` 按 `run.json` 递归发现，早先按一层遍历的版本会漏 run、也会在批次目录上崩。
- 整改前的整树备份在 `/data/shangguanrz/pic-edit-archive/workflow_runs_backup_20261012/`（含 `SHA256SUMS`）；审查与整改全过程见 `docs/workflow审查与整改_20261012.md`。
