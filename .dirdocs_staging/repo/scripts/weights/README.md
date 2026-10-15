# scripts/weights/ —— 跨架构权重继承：按审计白名单构造完整控制器的初始化 checkpoint
> 工作区路径：/home/shangguanrz/project/pic-edit/scripts/weights　|　本级：0 个子目录 / 1 个实体文件（`inherit_qwen21_full_weights.py`，9.8 KB，`ls -la` 2026-10-12 实测）　|　另含生成物 README.md、目录说明.md

## 一、这一层在干什么
只做一件事：把来源检查点（V6 训练产物）中语义可对应的张量，按显式白名单迁移进 Qwen21 完整控制器的接收检查点，产出 `--output` 初始化文件。它是"Qwen21 Full 线继承 V6 成果"这条工作线的唯一写入点，与 `scripts/audit/audit_qwen21_weight_compatibility.py`（先审计映射）、`scripts/run/run_qwen21_weight_inheritance.py`（继承后 60 图对照）、`scripts/reports/write_qwen21_inheritance_report.py`（出报告）构成 审计→转换→验证→报告 闭环。

## 二、层级地图
叶子目录，无子目录。

## 三、关键文件与实证内容
- `inherit_qwen21_full_weights.py`：argparse 必选项 `--source --receiver --whitelist --source-id --output`；"Create a full-controller initialization checkpoint from an audited whitelist"——未在 whitelist 的张量一律不迁移，来源用 `--source-id` 记录进元数据以便 provenance 复核。

## 四、注意事项 / 调用路径
- 用 qwen21 环境运行；执行前先跑权重兼容性审计，执行后跑 `run_qwen21_weight_inheritance.py` + `evaluate_qwen21_weight_inheritance.py`。
- 检查点实体在 `experiments/` 与 `/data/shangguanrz/pic-edit-archive/`，本目录不存放任何权重。
