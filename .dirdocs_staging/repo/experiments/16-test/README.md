# experiments/16-test/ —— 跨架构/跨条件 16 类高难编辑综合基准（DeepGen vs Qwen21 vs BFS-LoRA vs PEI2I）

> 路径：`/home/shangguanrz/project/pic-edit/experiments/16-test`　|　本级：13 个子目录 + 根脚本（`run_*.py` ×7、`derive_camera.py`、`render_ab_geometry.py`、`mk_prompts.py`）+ `prompts.json`/`prompts_16test.md`　|　子树：184 文件 / 85 MB / 94 张 PNG
> 数据来源：各 `stage*/results.json`、`prompts.json`，读取 2026-10-12

## 一、这一层在干什么
用最难的复合编辑场景（"让图一男士摆出图二姿势 + 手中托起一个篮球，保持外貌/深蓝西装/雨天城市街景不变，禁止引入图二的白影棚背景"）横向压测 6 条技术路线，回答一个战略问题：**单阶段端到端网络能否同时做到 大幅骨骼重定向 + 细粒度道具交互 + 面部 100% 冻结？** 结论是不能，必须走两阶段解耦（Adapter 定姿态构图 → BFS/Face 局部精修）。这是定性视觉对照套件，`results.json` 记录 `seconds/size/prompt/cfg/steps/seed/lora/adapter/condition` 等运行参数，**不含 PSNR/SSIM**（与 `qwen21_iper_body/four_arm_eval_50_s40` 的定量评测互补）。

## 二、层级地图（Stage 流水线 + 支撑目录）
| 子目录 | 装什么 | 内容实证 | 关系 |
| :--- | :--- | :--- | :--- |
| `stage1_baselines/` | Qwen2.1 基座纯 Prompt 直出 | `results.json` 4 例，qwen_base 160.14s/例、640×1088、cfg1.0、40 步；`qwen_base_grid.png` | 最弱基线 |
| `stage2_adapters/` | 挂 Adapter（deepgen/qwen × source/both） | 4 个 `results_*.json`（16 例），deepgen 用 `iper_v6_native_20k_v1/checkpoint_last.pt`、qwen 用 `full_20k/checkpoint_last.pt`；~36–41s/例，条件 `conditions_ab/pB (reframed to pA)` | 受控主力 |
| `stage3_detailed_prompt/` | 结构化拆解长提示词 | `results.json` 2 例 zh_detailed（身份/服装/背景负向排除） | Prompt 工程上限 |
| `stage4_bfs_body/` | 开源 BFS Body LoRA | `results.json` 7 例，lora `models/bfs/loras/bfs_body_swap_v1.0_qwen_2.1.safetensors`、strength 1.0、640×1088 | 面部/身体重定向对照 |
| `stage5_bfs_bracket_norm/` | BFS 官方归一化括号提示词 | `results.json` 3 例（a_norm_1024 84.37s/1024²） | BFS 标准化协议 |
| `stage6_pei2i/` | PEI2I 指令重写流 | `rewrites.json`、`rewrite_en.txt`、`rewrite_zh.txt` | 提示词重写对照 |
| `source/` | 输入源图 | `p-A/`（姿态参考）、`p-B/`（待编辑人物） | 全 stage 共享输入 |
| `conditions_ab/` | SMPL-X 几何条件渲染 | `pA/`、`pB/`、`camera.json` | Adapter 输入条件 |
| `stage2_cache/` | 条件潜变量缓存 | `16test_pA_pA.pt` | **stage2 依赖，不可删** |
| `_smoke/`、`_smoke_bfs/`、`_smoke_dgad/`、`_smoke_qwad/` | 各 runner 冒烟 | 少量图，分别验基座/BFS/DeepGen-Adapter/Qwen-Adapter 链路 | 冒烟，非结论 |

## 三、关键发现（跨 stage 定性结论）
1. `stage1` 纯 Prompt：基座完全无法按图二精确摆姿，易忘篮球或直接复制图二白影棚背景。
2. `stage2` Adapter：能强制肢体贴合图二姿势，但"托篮球"这种局部物理交互若无手/物细粒度条件，手掌与球易融合畸变。
3. `stage4/5` BFS LoRA：身体重定向背景保持较好，复杂几何约束仍依赖 Prompt 引导。
4. 战略结论：单阶段端到端无法兼顾三类目标 → **两阶段解耦**（Qwen21BodyAdapter 定姿态构图 → BFS/Face Retargeting 精修贴回）是当前最佳落地路线。

## 四、注意事项 / 调用路径
- 复现入口：根目录 `run_baseline_16test.py`、`run_deepgen_adapter.py`、`run_qwen_adapter.py`、`run_detailed_prompt.py`、`run_bfs_body.py`、`run_bfs_bracket_norm.py`、`run_pei2i.py`；条件构造 `render_ab_geometry.py` + `derive_camera.py`。
- `stage2_cache/16test_pA_pA.pt` 是 stage2 Adapter 复用的条件缓存，删除会触发重算。
- Adapter 权重分别来自 `iper_v6_native_20k_v1/checkpoint_last.pt`（DeepGen）与 `qwen21_iper_body/full_20k/checkpoint_last.pt`（Qwen）；BFS LoRA 在 `models/bfs/loras/`。
- 本套件是定性图证，勿与 `qwen21_iper_body/four_arm_eval_50_s40` 的定量数字混引。
