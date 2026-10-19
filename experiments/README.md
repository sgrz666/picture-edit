# experiments/ —— 人物姿态编辑全部实验的顶层索引与分组导航

`experiments/` 是 pic-edit 项目全部训练、评测、消融实验的落点。项目核心任务：保持源人物 A 的身份/服饰/背景不变，把 B 的姿态与表情重定向到 A 上。四条技术线在这里汇聚——① DeepGen-1.0-diffusers（2.47B MMDiT，冻结）+ 自研 Unified SMPL-X Adapter V6；② Qwen-Image-2.1（7B 单流 DiT）迁移线；③ workflow/ 两阶段基座流水线（无 adapter）；④ Qwen-Image-2.1 ControlNet Union 多条件控制线。

2026-10-13 16:30，原本平铺在 `experiments/` 下的 run 目录被重组为 5 个分组目录；2026-10-17 正式设立第 6 分组 `6.qwen-controlnet union/`，专门收拢 ControlNet Union 相关的全部代码与实验。本文件是重组后的总索引。

## 六个分组的职责与包含关系

| 分组 | 目录名 | 职责 | 内部分类 |
|---|---|---|---|
| 1 | `1.deepgen experiment/` | 技术线 1 全部 DeepGen 基座实验 | `champ/`、`iper/` 两个子分组 + 4 个散放 run |
| 2 | `2.qwen-test/` | Qwen 基座综合测试与原型 | `16-test`、`qwen-2.1-test`（注：union 迁移实验已归入第 6 族） |
| 3 | `3.singel man/` | Qwen 迁移线单人姿态控制 | 6 个 run（里程碑 `qwen21_iper_body` 等） |
| 4 | `4.two man /` | Qwen 迁移线双人交互与物理接触 | `hi4d_overfit_archive`、`qwen21_dual_prompt_pilot`、`qwen21_dual_v8`、`qwen21_hi4d_dual` |
| 5 | `5.workflow/` | workflow 两阶段基座流水线实验预留区 | 当前为空 |
| 6 | `6.qwen-controlnet union/` | Qwen 2.1 ControlNet Union 实验族（代码集中、模型软链加载） | `models/`、`qwen21_union_migration_v1`、`qwen21_union_dataset_adaptation_v1` |

分组内每个 run 目录均配有人工写的 `README.md`，可直接进目录阅读获取逐项事实。

## 中文名对照表

> 目录名本身未中文化：项目规则禁止改动已完成且已记录 `summary.json` 的实验目录名，以保证证据链完整（硬编码引用、冻结记录 `run.json`/`stage.json`、旧绝对路径都依赖原名）。下表仅作浏览对照。

**分组 1 · `1.deepgen experiment/`**
- `champ/champ_deepgen_formal_v1/` —— CHAMP 手势重定向正式运行（胸前比心 120 步）
- `champ/deepgen_no_control_ablation_v1/` —— 无控制消融对照（证姿态控制因果性）
- `iper/iper_v6_native_20k_v1/` —— V6 原生 20k 训练基准（4k 收敛拐点、16k 峰值 28.20 dB）
- `iper/iper_adapter_finetune_v1/` —— V6 共享条件库与 3k 微调（含 `condition_cache/`，不可删）
- `iper/iper_adapter_v1/` —— V6 适配器最早冒烟（100 步；**内含错嵌的 `iper_v65_face_overfit_024_6`**）
- `iper/iper_v6_native_finetune_v1/` —— V6 原生微调对照基线
- `iper/iper_v6_20k_eval_final/` —— V6.3 最终 50 样本评测（受控 +7.07 dB，dev-val 非 test）
- `iper/iper_v6_20k_eval_step4000/` —— V6 四千步评测与显存悬崖复现（OOM 中断）
- `iper/iper_v6_testdata_eval/` —— V6 测试数据评测（testdata +6.9 dB）
- `iper/iper_v64_detail_overfit_024_6/` —— V6.4 细节过拟合（iPER 024_6）
- `iper/iper_v65_face_overfit_smoke/` —— V6.5 面部过拟合冒烟
- `iper/iper_v65_face_overfit_024_6/`（错嵌于 `iper_adapter_v1/` 内）—— V6.5 面部过拟合（iPER 024_6，1500 步）
- `iper/iper_v6_comparative_3000/`（软链至归档盘）—— V6 三千步横向对照
- `iper/iper_test/` —— 512 分辨率直出测试
- `iper/iper_test_1024/` —— 1024 分辨率实验
- `tiktok_00001_condition_repair/` —— TikTok 条件修复对比（44 张图）
- `tiktok_objective_eval/` —— TikTok 客观基线评测（时间步扫描）
- `tiktok_overfit_sanity_full/` —— V6.5 面部门禁完整裁决（FAIL，推动路线转向）
- `v6_condition_inspection/` —— V6 条件投影检查

**分组 2 · `2.qwen-test/`**
- `16-test/` —— 十六条件综合基准（DeepGen/Qwen21/BFS/PEI2I 跨架构）
- `qwen-2.1-test/` —— Qwen21 双人交互原型（`data/testset/pairs/` 是 workflow 测试集来源）
- `qwen21_union_migration_v1/` —— （软链接指向 `../6.qwen-controlnet union/qwen21_union_migration_v1`，保持历史兼容）

**分组 3 · `3.singel man/`**
- `qwen21_iper_body/` —— Qwen21 身体控制主实验（里程碑，四臂评测 PSNR 25.86 dB）
- `qwen21_tiktok_v3_smoke/` —— Qwen21 身体+面部双适配器冒烟（TikTok v3）
- `tiktok_pe_prompt_rewrite_20261014/` —— TikTok 逐样本提示词重写（PE-I2I，新实验）
- `tiktok_v65_face_overfit_00001/` —— V6.5 面部过拟合（TikTok 00001）
- `tiktok_v65_pure_inference/` —— V6.5 面部开关纯推理对照
- `tiktok_v65_test_2steps/` —— V6.5 面部短步测试

**分组 4 · `4.two man /`**
- `hi4d_overfit_archive/` —— Hi4D 拥抱过拟合归档（hug 120/1000 步）
- `qwen21_dual_prompt_pilot/` —— 双人提示词（short vs structured）对照 Pilot
- `qwen21_dual_v8/` —— dual v8 双人适配器设计验证
- `qwen21_hi4d_dual/` —— Qwen21 双人交互与物理接触（Hi4D）

**分组 5 · `5.workflow/`**：空，预留给 workflow 相关实验。

**分组 6 · `6.qwen-controlnet union/`**
- `models/` —— 统一软链接区（Qwen 2.1 7B 基座与 ControlNet Union 7.1G 权重，零物理拷贝）
- `qwen21_union_migration_v1/` —— 官方 Union 控制网直接迁移验证与小样本评估（原位于 2.qwen-test，已归位）。三关全过；48 张矩阵里接入版 scale 0 与现工作流逐字节相同（24/24 配对），单人严格 PCK@0.2 0.750→0.964 但 scale 1.0 身份分 0.916→0.585 且骨架线漏进画面、双人检出人数并成 1 人 → 判定「需要进一步输入适配」。注：GitHub 仓库内实际落盘路径是平铺的 `experiments/qwen21_union_migration_v1/`，分组目录结构只在服务器侧生效
- `qwen21_union_dataset_adaptation_v1/` —— 多数据集（iPER/TikTok/Hi4D）LoRA 联合适配训练实验（代码仓内管理，数据与产物软链落盘）

## 关键资产路径（不可随意删除）

- `experiments/3.singel man/qwen21_iper_body/full_20k/checkpoint_best.pt`（约 772 MB，已核实存在）——项目最强身体控制权重，被 `qwen21_hi4d_dual` 继承为双人训练初始化。
- `experiments/1.deepgen experiment/iper/iper_adapter_finetune_v1/condition_cache/`（584 个 iPER SMPL-X 特征张量，约 2 GB，已核实存在）——V6 系列训练与推理的共享底层依赖，**不可删**。
- `experiments/1.deepgen experiment/iper/iper_v6_native_20k_v1/checkpoint_best.pt`（约 1.59 GB，已核实存在）——V6 单人基准 best 权重。
- `experiments/2.qwen-test/qwen-2.1-test/data/testset/pairs/`（hard_01~05 + medium_01/02 共 7 组源-目标对，已核实存在）——workflow 测试集 7 对的来源；`workflow/batch.sh` 与 `workflow/report.py` 依赖此路径。
- `experiments/6.qwen-controlnet union/models/Qwen-Image-2.1-Fun-Controlnet-Union.safetensors`（7.1 GB，已通过 180 键完整性校验）——官方 ControlNet Union 单模型多条件控制网络核心权重。

## 实验创建规范

新建实验目录命名：`{主题}_{YYYYMMDD}`（如 `tiktok_pe_prompt_rewrite_20261014`）或 `{主题}_v<N>`（如 `iper_adapter_finetune_v1`）。**禁止**使用 `test`、`tmp`、`new` 等无意义/临时词作目录名。每个 run 目录应记录 `summary.json` / `run_manifest.json` 等结构化证据，已完成实验目录严禁改名。

所有 Qwen ControlNet Union 相关的实验、代码均统一收拢于 `6.qwen-controlnet union/`，模型权重一律软链接加载，严禁把 GB 级权重实体考入代码仓库。
