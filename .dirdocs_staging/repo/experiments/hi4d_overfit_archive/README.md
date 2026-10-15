# experiments/hi4d_overfit_archive/ —— DeepGen 时代 Hi4D 双人拥抱过拟合归档

> 路径：`/home/shangguanrz/project/pic-edit/experiments/hi4d_overfit_archive`　|　2 个子目录 + 直属文件　|　子树：58 文件 / 23 MB / 40 张 PNG　|　数据源：各 `summary.json`、`experiment_config.json`，2026-10-12

## 一、这一层在干什么
在转向 Qwen-Image-2.1 双人线之前，用 **DeepGen-1.0** 基座在 Hi4D 单条双人样本（pair01/cam4，hug01→hug01 同动作、源帧未接触→目标帧接触）上做的小规模过拟合归档。为后续 Qwen 双人线（`qwen21_hi4d_dual/`）提供"接触条件渲染 + 双人提示词"的经验来源与 `hi4d_dual_source.pt` 条件缓存。

## 二、层级地图
| 子目录 | 装什么 | 内容实证 | 关系 |
| :--- | :--- | :--- | :--- |
| `hug_120steps/` | 120 步过拟合 | `summary.json` completed_steps=120、duration 186s；diagnostics step30/60/90/120；`model_step_0000…0120.png` + `comparison_grid.png`；`condition_cache/` | 早期短程 |
| `hug_1000steps/` | 1000 步过拟合 | `summary.json` completed_steps=1000、duration 1045.6s；loss step100 0.073→step1000 0.034；11 张 `model_step_*.png` | 取代 120 步（更长程） |

## 三、关键文件与实证
- `experiment_config.json`：`manifest=Hi4D_pilot_v1/final_assets_v4_2/manifests/main_pairs_train.jsonl`, row=0, seed=42, inference_steps=24, max_steps 分别 120/1000；prompt 为双人拥抱保真提示词。
- `summary.json.row`：`dataset_bucket=MAIN_N2C`、`pair=pair01`、`camera_id=4`、`pair_relation=SAME_ACTION`、source 未接触→target 接触（`target_sampling_case=0C_3S`）。
- 条件渲染图：`02_CONDITION_{POSE,DEPTH,NORMAL,MASK}_{A,B}.png`、`01_original_deepgen_language_only.png`、`00_SOURCE/TARGET_IMAGE.png`、`03_LANGUAGE_EDIT_PROMPT.txt`。
- `hug_1000steps/condition_cache/hi4d_dual_source.pt`：双人源条件缓存（下游 Qwen 双人线沿用其构造思路）。

## 四、注意事项
归档目录，DeepGen 基座产物，**非当前权威**（当前双人方向在 `qwen21_hi4d_dual/`、`qwen21_dual_v8/`）。仅 120/1000 步单样本过拟合，无泛化含义。
