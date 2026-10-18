# scripts/weights/ —— 权重继承与转换

本目录是系统的「权重初始化 / 转换」层，专门处理可训练 controller 的**初始化 checkpoint 构造**，而不涉及训练本身。它在 `scripts/audit/`（权重溯源、白名单审计）与 `scripts/train/`（真正训练）之间补齐一环：把经过审计的权重来源按白名单组装成一个干净的初始化 checkpoint，供后续训练入口加载。父级 [`../README.md`](../README.md) 把 `weights/` 定位为「权重继承和转换」。

## 里面有什么

| 文件 | 一句话职责 |
|---|---|
| `inherit_qwen21_full_weights.py` | 从**审计过的白名单**创建一个 full-controller 的初始化 checkpoint。脚本自身只读取被批准的权重来源并把它们组装成初始化权重，不做训练、不修改源权重 |

## 与上下层的关系

- **上游输入**：`scripts/audit/audit_qwen21_weight_compatibility.py` 与 `inventory_qwen21_full_weights.py` 产出权重溯源与逐张量白名单；`models/`（Qwen-Image-2.1、DeepGen）提供基础张量。
- **产出物**：一个 full-controller 初始化 `.pt`（或等价 checkpoint），被 `scripts/train/train_iper_qwen21_full.py` 等完整 controller 训练入口作为起点加载。
- **下游验收**：`scripts/audit/check_qwen21_full_preservation.py`、`verify_qwen21_full_resume.py` 等会核对这个初始化 checkpoint 的来源与可复现性；其权重审计证据由 `scripts/reports/write_qwen21_weight_audit_report.py`、`write_qwen21_inheritance_report.py` 写成报告。
- 相关索引：父级 [`../README.md`](../README.md)；权重/资产路径登记在 [`../../configs/asset_registry.json`](../../configs/asset_registry.json)。

## 注意与坑

- 本目录只有 1 个脚本，职责很窄：**只做初始化权重的「继承/组装」，不训练、不重写源权重**。改白名单前应先跑对应审计脚本确认来源合法。
- 生成的初始化 checkpoint 是训练的**起点**而非训练结果；它与 `scripts/train/` 产出的真正训练 checkpoint 要区分开，避免把「继承初始化」误读成「已训练权重」。
- 该脚本的依赖与 `scripts/audit/` 的权重审计强耦合——白名单/来源一旦变动，需重新审计后再生成初始化权重。
