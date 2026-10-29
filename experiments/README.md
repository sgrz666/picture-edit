# experiments/ —— 项目全量实验运行数据与资产导航

> **核心原则**：
> 1. 本目录记录模型从早期显存压测、DeepGen MMDiT V6 系列、Qwen-Image-2.1 单流 DiT 身体/面部/双人交互适配器，到 Qwen 2.1 Pro 与 Native V5 体系的全部实验运行（Run）。
> 2. 大体积检查点（`.pt` / `.safetensors`）与中间条件缓存持久化存储于独立数据盘 `/data/shangguanrz/pic-edit-experiments/`。
> 3. 服务器 `sg` 远端按技术线分为分组目录；本地工作区主要保留关键轻量配置、脚本与指标汇总。

---

## 一、核心实验族与物理落点导航总表

| 核心实验族 | 关键代表目录 | 物理落点 (`/data` 或仓库) | 核心架构与模型 | 关键结论与状态 |
|---|---|---|---|---|
| **1. 显存与架构压测族** | `review_20260908_v2` | 归档于 `/data/shangguanrz/pic-edit-archive/` | DeepGen-1.0 (2.47B MMDiT) | 111 个运行矩阵 100% PASS；确立 Letterbox 显存安全标准与 80G 并发容量 |
| **2. DeepGen V6 单人长程** | `iper_v6_native_20k_v1` | `experiments/1.deepgen experiment/iper/` | DeepGen-1.0 + V6 Unified Adapter | 4k 步达关键跃迁；16k 步达保真峰值（PSNR 28.20dB, SSIM 0.9361） |
| **3. DeepGen 定量与消融** | `iper_v6_20k_eval_final` | `experiments/1.deepgen experiment/iper/` | DeepGen-1.0 + V6 系列 | 50 样本评估：受控相比无控制 PSNR 提升 +7.07 dB，误差降低 57.2% |
| **4. 面部适配器与过拟合** | `tiktok_overfit_sanity_full` | `experiments/1.deepgen experiment/tiktok_*` | DeepGen-1.0 + V6.5 Face | 严格证明 MMDiT 单层特征注入中晚期破坏背景全局流（门禁FAIL），推动解耦路线 |
| **5. Qwen 2.1 身体控制族** | `qwen21_iper_body` | `experiments/3.singel man/` | Qwen-Image-2.1 (7B DiT) + BodyAdapter | 里程碑：PSNR 25.86dB (+1.33dB)，PCK@0.02 达 98.00%，突破单人身体控制 |
| **6. Qwen 2.1 双人交互族** | `qwen21_hi4d_dual` | `experiments/4.two man /` | Qwen-Image-2.1 + DualAdapter | 双人检测率 100%，零身份互换；拥抱过拟合 PSNR 达 29.85dB (+17.5dB) |
| **7. 综合评测与开源对比** | `16-test` | `experiments/2.qwen-test/` | DeepGen vs Qwen vs BFS LoRA | 16 组全场景对照，确立“Stage 1 Adapter 控姿 + Stage 2 局部精修”两阶段架构 |
| **8. ControlNet Union 控制** | `6.qwen-controlnet union` | `/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1` | 官方 Qwen-Image-2.1-Fun-Controlnet-Union (7.1GB) | 180 权重键全部通过审计；TikTok 8 对实测：对齐布局下 PCK@0.2=0.976 打平基线；纯控制模式 ID 降至 0.01~0.08 |
| **9. Qwen 2.1 Pro 适配器** | `qwen21_pro_adapter_v1` | `/data/shangguanrz/pic-edit-experiments/qwen21_pro_adapter_v1` | 官方 16 个原生控制块 (4096维) + 14.6M FP32 ProTaskAdapter | 零初始化安全相加；独立外观 Bank (`visible_owner`)；200步过拟合门禁：Face loss -31.86%, Contact -64.37%, Hand -29.44% |
| **10. Native V5 固定样本训练** | `native_v5_fixedpair` | `/data/shangguanrz/pic-edit-experiments/` (tiktok & hi4d fixedpair-v5) | Qwen 2.1 Native V5 模块化全量训练架构 | 单人 (TikTok 0374/0126) 与双人 (Hi4D 0082/0112) 300 步训练；全套审计与无检测器 ROI 过程观察 |

---

## 二、最新重点实验物理落点与产物说明

### 1. Qwen 2.1 Pro 适配器实验区 (`/data/shangguanrz/pic-edit-experiments/qwen21_pro_adapter_v1/`)
- 源码核心：[`src/pose_control/qwen21_pro/`](../src/pose_control/qwen21_pro/)。
- 运行入口：[`src/pose_control/qwen21_pro/runner.py`](../src/pose_control/qwen21_pro/runner.py)（支持 `prepare`, `cache`, `audit`, `smoke`, `train`, `infer`, `refine`, `status`）。
- 关键产物：
  - `runs/pilot_main200_v4_20261018T161055/checkpoint_000200/`：200 步完整训练检查点（包含优化器、LR 调度器与 RNG 状态）。
  - `runs/cli_smoke_exact_20261018T154445/`：2+2 步恢复与 4 步连续逐位一致性（Bit-exact）审计记录。
  - `runs/status_report_20261018T170600/`：门禁评估与停机证据报告。

### 2. Native V5 固定样本训练区 (`/data/shangguanrz/pic-edit-experiments/`)
- TikTok 单人落点：`qwen21_full_native_v1/tiktok-fixedpair-v5_20261025_v1/`。
- Hi4D 双人落点：`qwen21_hi4d_dual/hi4d-fixedpair-v5_20261025_v1/`。
- 关键特性：每 30 步导出过程图、包含基础 Body (S0) 与局部路径 (Face/Hand/Source Detail) 消融对比。
- 报告与可视化工具：[`scripts/reports/export_qwen21_fixedpair_report.py`](../scripts/reports/export_qwen21_fixedpair_report.py) 与 [`reports/native_v5_fixedpair_20261026_v1/`](../reports/native_v5_fixedpair_20261026_v1/README.md)。

### 3. TikTok 官方 Union 实测 (`workflow/runs/tiktok/union-test/`)
- 部署与产物：`workflow/runs/tiktok/union-test/`。
- 关键数据：8 对样本 4 臂横向评测（T1, T2b, D1, D2b 共 32 组推理），证实骨架对齐到 A 时 PCK@0.2 达到 0.976（与 Baseline 8/8 逐对一致），但在不传 A 图时 ID 余弦仅 0.012~0.087。
