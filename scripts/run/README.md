# scripts/run/ —— 实验编排与收尾运行入口

本目录是系统的「一键运行 / 编排」层：把 `scripts/train/` 的训练、`scripts/eval/` 的评测与 `scripts/data/` 的补数据步骤串成可重复的实验流程，并提供 demo、消融、单样例诊断与事后收尾脚本。它直接驱动 `experiments/<run>/` 下的真实产物（图像、checkpoint、日志），因此多数为有 GPU/有副作用的入口；少数 `.sh`/`.ps1` 只是带环境的启动器。父级 [`../README.md`](../README.md) 把 `run/` 定位为「实验编排与收尾」。

## 里面有什么

### DeepGen 演示 / 消融 / 单体过拟合

| 文件 | 一句话职责 |
|---|---|
| `run_demo_pose_edit.py` | 基于 DeepGen-1.0-diffusers 的单图姿态编辑 demo |
| `run_deepgen_no_control_ablation.py` | 纯 DeepGen 消融：不注入 adapter 或空间控制，作为基线 |
| `run_champ_single_overfit.py` | 在单个对齐 CHAMP source/target pair 上过拟合 PoseConditionAdapter |
| `run_tiktok_sample_00001_smoke.py` | TikTok 序列 00001（Source 0014→Target 0074）的冒烟：校验条件加载、V6.5 Face 特征对齐、前反向梯度流、VRAM<24GiB、预览导出 |

### Qwen21 运行入口

| 文件 | 一句话职责 |
|---|---|
| `run_qwen21_weight_inheritance.py` | 不训练，验证并生成固定的 60 图权重继承对照 |
| `run_qwen21_condition_combinations.py` | 跑两个锁定的正式推理批次，带持久化阶段状态 |
| `run_qwen21_face_single_sample.py` | 有界单 pair 诊断，每个 GPU 动作都用完整 controller（被 `scripts/audit/probe_qwen21_face_single_sample.py` 依赖） |
| `run_qwen21_joint_pilot.py` | 有界 joint Body+Face pilot；先看已保存图像再批准 mid-stage 续跑 |

### 收尾 / 完成

| 文件 | 一句话职责 |
|---|---|
| `finalize_qwen21_face_repair.py` | 汇总已保存 Face-repair 检查与人工复核的单样例 |
| `finalize_qwen21_face_single_sample.py` | 收集有界运行结果并核对显式单样例标准 |
| `complete_hi4d_qwen21_dual.py` | 跑完长时 Qwen21 dual 运行，再评测留出的 Hi4D test split |

### Shell / PowerShell 启动器

| 文件 | 一句话职责 |
|---|---|
| `run_tiktok_v3_fill_missing_geometry_bg.sh` | 后台跑 TikTok V3 缺失几何补全（用 `add/envs/smplestx_bw` 环境） |
| `run_v6_comparative_3000.sh` | V6.3 对照训练启动器：在精确基线样本流上跑 3000 微步到 `experiments/iper_v6_comparative_3000`（deepgen 环境） |
| `sync_qwen21_joint_pilot.ps1` | 同步 joint pilot 产物的 PowerShell 脚本 |

### `home/` 子目录 —— 统一运行入口

`scripts/run/home/` 不是脚本集合，而是 **4 个指向宿主目录 `/home/shangguanrz/` 正式 HOME runner 的软链接**，由父级 `scripts/README.md` 的整理说明明确登记（「`run/home/` 另有四个正式 HOME runner 的链接；原 HOME 脚本继续保留」）：

| 软链接 | 指向 | 作用 |
|---|---|---|
| `run_smplestx.sh` | `/home/shangguanrz/run_smplestx.sh` | 调 `add/tools/SMPLest-X` 的 `inference.py`，对 pA/pB 做 SMPL-X 拟合（不 `--retarget_cam`，保持导出相机与照片一致），回收 `*_params.pt` |
| `run_stage1.sh` | `/home/shangguanrz/run_stage1.sh` | 16-test 第一阶段：分别用 deepgen / qwen21 两套 conda 环境跑 `run_baseline_16test.py`（deepgen_base、qwen_base）到 `experiments/16-test/stage1_baselines` |
| `run_stage2_both.sh` | `/home/shangguanrz/run_stage2_both.sh` | 16-test 第二阶段：在 `--refs both` 变体上跑 DeepGen/Qwen adapter，隔离 adapter 净贡献，输出到 `stage2_adapters` |
| `watch_and_run.sh` | `/home/shangguanrz/watch_and_run.sh` | 等待指定 eval 进程退出后，自动 nohup 启动 `run_baseline_16test.py` 的 stage1 基线 |

这 4 个脚本对应的是 **`workflow/` 两阶段基座流水线**（A 保外观、B 供动作，SMPLest-X→DWPose 骨架，Qwen-2.1 基座生成）在宿主机器上的统一运行入口，不属于带 adapter/LoRA 的训练线。它们指向项目外的宿主路径，**整理时未复制源码，仅保留链接**。

## 与上下层的关系

- **上游输入**：`scripts/train/`（训练逻辑）、`scripts/eval/`（评测逻辑）、`scripts/data/`（补数据）、`models/`、`configs/asset_registry.json`（资产路径）。
- **产出物**：`experiments/<run>/`（checkpoint、采样图、日志）、`results/`、`logs/`；部分运行脚本调用的评测/收尾结果再喂给 `scripts/reports/`。
- **下游消费者**：`scripts/audit/` 的 `verify_qwen21_full_runtime.py`、`verify_qwen21_face_single_sample.py` 等验收这些运行的可复现性；`scripts/reports/` 消费其产物。
- 相关索引：父级 [`../README.md`](../README.md)；训练线细节见 [`../train/README.md`](../train/README.md)；workflow 两阶段基座见仓库 `workflow/`。

## 注意与坑

- `home/` 下是**项目外宿主脚本的软链接**，不在仓库内维护；改这些运行逻辑要去 `/home/shangguanrz/` 改原文件，`scripts/run/home/` 只是入口映射。
- `run_*` 多数为有副作用入口（写 `experiments/`、占 GPU），跑前确认目标 run 目录与 GPU 状态；`run_v6_comparative_3000.sh` 等 `.sh` 已写死各自 conda 环境路径。
- `run_qwen21_face_single_sample.py` 与 `scripts/audit/probe_qwen21_face_single_sample.py` 通过共享常量耦合，改动输出格式需两边同步。

## Native Body21 推理

`run_qwen21_native_pose.py` 读取独立 runtime checkpoint、目标几何缓存与源 RGB，支持 TikTok/iPER 和可选 typed Depth；KV cache 关闭，参考预算显式记录。`--preflight-only` 只核验条件与 checkpoint，不加载 Qwen 或采样。输出必须使用新目录；本轮没有训练更新或采样质量结论。详见 `src/pose_control/qwen21_full/GLOBAL_GEOMETRY_V2.md`。

## Depth V2 接线（未运行验收）

Native推理按checkpoint版本原样重建，`--depth-version`仅作断言，`--depth-mode auto`只读取manifest明确声明的模态。V2使用同一Body21记录、23通道几何和29维Gate上下文；不会从PNG猜Z。V1保留。当前未运行模型或采样，后续命令见 `src/pose_control/qwen21_full/NATIVE_DEPTH_V2.md`。

## Native Dual V3（尚未运行验收）

`run_qwen21_native_dual.py` 使用独立V3 checkpoint和同源双人manifest，仅加载源RGB，直接读取实际Qwen参考latent前缀完成身份绑定。保持一张参考图、B=1和原Native Body，不新增Qwen token。输出须新目录，详见 `src/pose_control/qwen21_full/NATIVE_DUAL_V3.md`。
