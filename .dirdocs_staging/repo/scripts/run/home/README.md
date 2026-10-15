# scripts/run/home/ —— 四个 HOME 正式 runner 的分类入口（全部为符号链接）
> 工作区路径：/home/shangguanrz/project/pic-edit/scripts/run/home　|　本级：0 个子目录 / 4 个符号链接 / 0 个实体脚本（`ls -la` 2026-10-12 实测）　|　另含生成物 README.md、目录说明.md

## 一、这一层在干什么
2026-10-03 目录整理时，HOME（`/home/shangguanrz/`）下混着正式 runner 与 62 份历史诊断脚本。整理把历史工具逐字节归档到 `archive/home_tools_202609/`，四份仍被日常使用的正式 runner 保留 HOME 实体（脚本内含 `$0`/dirname 路径依赖，已核实不宜迁移），本目录只为它们提供"可从仓库内找到"的分类入口。

## 二、层级地图
| 链接 | 实体 | 干什么 |
|---|---|---|
| `run_smplestx.sh` | `/home/shangguanrz/run_smplestx.sh` | 驱动 `add/tools/SMPLest-X` 对 pA/pB 两张图逐帧拟合 SMPL-X（明确不带 `--retarget_cam` 以保持相机与原图对齐，否则 mesh_cam 无法对齐照片）；用 `add/envs/smplestx_bw` 环境 |
| `run_stage1.sh` | `/home/shangguanrz/run_stage1.sh` | `experiments/16-test` 阶段一基线：DeepGen 与 Qwen 两个基干必须各用自身 conda 环境（transformers 大版本不同），输出 `stage1_baselines/` |
| `run_stage2_both.sh` | `/home/shangguanrz/run_stage2_both.sh` | 阶段二 adapter 对照：等待在跑的 `--refs source` 结束后补跑 `--refs both` 变体（与基干同参考图、隔离 adapter 净贡献），输出 `stage2_adapters/` |
| `watch_and_run.sh` | `/home/shangguanrz/watch_and_run.sh` | 等指定 eval PID 退出、显存释放后再 nohup 启动 16-test 基线（GPU 排队器；硬编码的 PID 2790689 属一次性使用） |

## 三、注意事项
- 本级没有任何实体代码，改动逻辑请编辑 HOME 下原文件；仓库内看到的"同名脚本"只是链接。
- 62 份旧 HOME 工具在 `archive/home_tools_202609/`（含独立 `operations.json`），仅供查阅，勿当运行入口。
