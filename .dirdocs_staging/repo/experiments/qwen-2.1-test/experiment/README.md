# experiments/qwen-2.1-test/experiment/ —— Qwen2.1 双人/单人编辑探索代码与结果

> 路径：`…/qwen-2.1-test/experiment`　|　子目录 `code/`、`results/`；`prompts_gen_v1.md`　|　数据源：`results/overfit_inter01/metrics.json`、各 `results.json`，2026-10-12

## 一、干什么
本验证台的实验主体：独立的探索代码（`code/`）+ 各类生成/过拟合产物（`results/`）。回答"Qwen2.1 靠提示词+SMPL-X 几何能否做出双人拥抱、单人姿态迁移"。

## 二、层级地图
| 子目录 | 装什么 | 内容实证 |
| :--- | :--- | :--- |
| `code/` | 探索脚本（脱离主 `src/`） | `overfit_inter01.py`、`hi4d_conditions.py`、`qwen21_body_adapter_dual.py`、`test_conditions.py` |
| `results/overfit_inter01/` | 单样本双人过拟合 | `metrics.json`（off 恒 12.354，on 300 步 27.40 / 400 步 30.14 / 500 步 29.85, SSIM 0.9245）、`train_log.json`、`input_conditions.json`、26 图 |
| `results/interaction_v1/` | 双人 12 条 case 详细臂 | `results.json`（inter_01 seconds 122.0, peak 19.0 GiB, 768×1024）、39 图 |
| `results/gen_v1/` | 单人姿态迁移多 case | `results.json`（hard_01 等）、21 图 |
| `results/smoke/`、`results/overfit_smoke/`、`results/overfit_inter01_cache/` | 冒烟与中间缓存 | 少量图 / `train_log.json` / 空 |

## 三、注意
`overfit_inter01` 高 PSNR 为**同条样本重建**、非泛化；代码与主 `src/` 可能分叉，正式实现见 `src/pose_control/qwen21/`。数据源在兄弟目录 `../data/`。
