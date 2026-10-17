# PE-I2I GGUF 延迟审计（2026-10-16）

核对本地及 sg 的 GGUF 选型实验，解释为何量化后每对仍需数分钟，并为“模型常驻后每次重写 ≤20 秒”划出可检验的优化顺序。本目录保存只读审计，不运行模型推理或训练。

## 里面有什么

- `collect_readonly.py`：读取 sg 上既有 GGUF/HF 记录、服务日志、生产代码、已安装二进制帮助与 CPU tokenizer，向标准输出写 JSON。只读文件，不启动 llama-server，不分配模型 GPU 张量。
- `source/`：从 sg 复制的历史日志、逐次结果、提示词及代码快照。复制不改变原件；不含模型权重。
- `audit.json`：本次实际执行的时间、代码 SHA、原始结果、统一 token 口径、服务计时与二进制支持的开关。以文件中的采集时间为准。
- `report.md`：实验清单、慢的原因、旧统计需要纠正的地方，以及 20 秒目标的验证步骤。

## 已核对的关键数字

现有独立 GGUF 实验只有 Q8_0＋MTP 的一次 smoke 和两对正式旁路比较；Q4_K_M 仅做过元数据/社区证据调研，没有本机推理结果。两对 GGUF 含重试共 987.2 秒，BF16 共约 954.0 秒；完整计算量分别为 19,380 与 13,637 个输出 token。不能只报成功答案的 token 数来计算含重试吞吐。

GGUF `seconds_total` 的计时在 server ready 之后，已经排除模型冷加载；常驻不会消除这 987.2 秒中的开销。正式单人 worker 当前仍用 HF BF16。两份 GGUF 脚本均带 `--no-mmproj-offload`；本机 b11617 的帮助同时确认存在 `--reasoning-budget`。20 秒是待验证目标，本次没有测得该延迟。

## 复核与上下游关系

上游是 `experiments/pe_i2i_gguf_selection_20261014/`、sg 的同名 `/data/shangguanrz/pic-edit-experiments/` 目录，以及 TikTok PE 吞吐消融。延续 10 月 14 日量化选型和 10 月 15 日蒸馏审计，不更改历史实验、生产 worker、运行中的 PE bank、环境或模型。

脚本用 `scp` 直传到 `/data/shangguanrz/pe_i2i_gguf_latency_audit_20261016_<时间戳>/`，先语法检查，再用既有 qwen21 Python 执行。JSON 保存到新文件，不覆盖前次快照。后续 GPU 对照应归到既有 GGUF 实验目录，用新 output 名称，并等独立资源窗口；详细设置与计时口径见 `report.md`。
