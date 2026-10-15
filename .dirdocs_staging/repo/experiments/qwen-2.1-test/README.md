# experiments/qwen-2.1-test/ —— Qwen-Image-2.1 双人/单人编辑的早期手工验证台

> 路径：`/home/shangguanrz/project/pic-edit/experiments/qwen-2.1-test`　|　2 个子目录（`data/`、`experiment/`）+ `test.txt`(空)　|　子树：435 文件 / 1.1 GB / 237 张 PNG　|　数据源：`experiment/results/overfit_inter01/metrics.json`、各 `results.json`，2026-10-12

## 一、这一层在干什么
项目早期在 Qwen-Image-2.1 上做双人（Hi4D 太极→拥抱）与单人（姿态迁移）编辑可行性的**手工验证台**：自带一份小测试数据（`data/`）和独立于主 `src/` 的探索代码（`experiment/code/`），跑单样本过拟合与多 case 生成，回答"基座能不能靠提示词+几何做到双人拥抱"。它是 `qwen21_hi4d_dual/` 正式双人适配器线的前身灵感来源。

## 二、层级地图
| 子目录 | 装什么 | 内容实证 | 关系 |
| :--- | :--- | :--- | :--- |
| `experiment/` | 探索代码 + 生成/过拟合结果 | `code/`（`overfit_inter01.py`、`hi4d_conditions.py`、`qwen21_body_adapter_dual.py`、`test_conditions.py`）；`results/`：`overfit_inter01/`、`interaction_v1/`(39 png)、`gen_v1/`(21 png)、`smoke/`、`overfit_smoke/` | 实验主体 |
| `data/` | 该验证台的自备测试数据 | `testset/`（`manifest.json` + `pairs/hard_01…05`,`medium_01/02` + `A_heavy`,`B_slim`）、`interaction/testset/inter_01…12`（taichi/hug/dance/backhug/pose 12 条）、`candidates/`（heavy/slim/_rejected/contact_sheets） | 供 `experiment/` 消费 |

## 三、关键文件与实证
- **`experiment/results/overfit_inter01/metrics.json`**（单样本 inter_01_pair00_taichi，源 taichi00#41→目标 hug00#108, cam4）——过拟合演变的权威曲线，`off`（无控制）全程恒为 **PSNR 12.354 / SSIM 0.40093**（原生基座做不到），`on`（几何控制）随步攀升：step0 12.354 → step100 16.919 → step200 19.683 → step300 27.401 → step400 **30.142** → step500 29.846（SSIM 0.92449）。即顶层引用的"+17.5 dB、12.35→29.85"结论，但**这是同一条样本的重建过拟合，非泛化**。
- `input_conditions.json`：几何张量清单——`pose_heatmap_a`(1,25,512,512)、`part_onehot_a`(1,14,512,512)、`normal_a`(1,3,512,512) 等，task_id=2/HUG。
- `results/interaction_v1/results.json`：双人 12 条 case 的 detailed 臂生成，`inter_01` seconds 122.0、peak 19.0 GiB、输出 768×1024、n_refs=1。
- `results/gen_v1/`、`smoke/`、`overfit_smoke/`：单人姿态迁移与冒烟生成，含 prompt/耗时（`hard_01` 等）。

## 四、注意事项
- 本目录是**早期探索台**，`experiment/code/` 与主仓库 `src/` 可能分叉；正式实现以 `src/pose_control/qwen21/dual_adapter.py`（legacy_dual_lite）与 `src/pose_control/qwen21_full/dual/`（v8）为准。
- `overfit_inter01` 的高 PSNR 只在训练样本成立，评测须与 `qwen21_hi4d_dual/single_*` 审计的诚实结论对照：开双人适配器后 DWPose 多检为 1 人，双人指标不可靠。
- `data/` 的 12 条 interaction、若干 pairs 为小测试集，非 Hi4D 全量。
