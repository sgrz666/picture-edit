# scripts/ —— workflow 私有工具目录

scripts/ 放「只服务 workflow 流水线」的辅助工具，与仓库级 `scripts/`（面向单人 / 双人两条线共用的通用能力）分工。每个工具自包含：自带 `script/`（源码）、`output/`（产物）、`README.md`（用法），符合 AGENTS.md 规定的工具目录标准。

## 里面有什么

- `six-picture-create/`：把一次 workflow run 的六件关键产物（A 源图 / B 目标图 / B 骨架拟合 / 对齐后姿态引导 / 引导叠 A / 最终生成图）拼成一张带标注的图，用于人工审图。结构 `script/six_picture_create.py` + `output/*.png` + 自带 `README.md`；支持 `--all` 批量、`--check` 只校验、`--fallback` 借基线 run 补消融 run 缺的面板。是 AGENTS.md 工具目录标准的样板。

## 与上下层的关系

- 消费正式 run：默认读 [`../runs/`](../runs/README.md) 下各 run 产物（如 `tiktok` 的六图面板软链到本目录 `output/tiktok/`）。
- 与仓库级 `scripts/` 的区别：仓级 `scripts/` 是两条线共用的通用能力；本目录只服务单人 workflow，不放跨线或训练侧代码。
- 流水线总说明见 [`../README.md`](../README.md)；运行产物见 [`../runs/README.md`](../runs/README.md)。

## 注意与坑

- 新增工具请严格按 `six-picture-create` 的样板建 `script/`、`output/`、`README.md` 三件套，别把脚本直接丢在 scripts/ 根。
- `output/` 是工具自己出的图，不是流水线证据链的一部分；清理时可与 run 产物区分对待。
- 工具读取的元数据来自 `geometry/stage.json` 的 `metadata.camera_checks`（scale / canvas / valid joints），换过阶段 schema 的 legacy run 可能取不到，需 `--fallback`。
